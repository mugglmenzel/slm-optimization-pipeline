# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Authors:
# - Dr. Jens Kohl, jens.kohl@bmw.de
# - Dr. Michael Menzel, michaelmenzel@google.com
# - Marcel Gotza, mgotza@google.com

"""A Kubeflow pipeline for building, optimizing, and evaluating SLMs for automotive use cases on Vertex AI."""

import os
from datetime import datetime
from typing import List

from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.types.artifact_types import (
    UnmanagedContainerModel, VertexModel)
from kfp import compiler, dsl


# ==============================================================================
# INDIVIDUAL, REUSABLE OPTIMIZATION & COMPRESSION COMPONENTS
# ==============================================================================


@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform>=1.49.0",
        "google-generativeai",
    ],
)
def gemini_supervised_finetuning(
    project: str,
    location: str,
    model_display_name: str,
    finetuning_dataset_uri: str,
    base_gemini_model: str,
    model: dsl.Output[VertexModel],
) -> None:
    """Performs supervised fine-tuning of a Gemini model using the GenAI SDK."""
    import logging

    from google import genai
    from google.cloud.aiplatform_v1.services import model_service
    from google.genai.types import CreateTuningJobConfig, HttpOptions, TuningDataset
    from google.protobuf import json_format

    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Starting Gemini supervised finetuning for model '{base_gemini_model}' "
        f"with dataset: {finetuning_dataset_uri}"
    )

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    tuning_job = client.tunings.tune(
        base_model=base_gemini_model,
        training_dataset=TuningDataset(gcs_uri=finetuning_dataset_uri),
        config=CreateTuningJobConfig(tuned_model_display_name=model_display_name),
    )
    logging.info(f"Submitted tuning job: {tuning_job.name}. Waiting for completion...")

    # .result() is a blocking call that waits for the job to finish.
    tuning_job.result()

    tuned_model_name = tuning_job.tuned_model.model
    logging.info(f"Finetuning complete. Tuned model resource name: {tuned_model_name}")

    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    model_client = model_service.ModelServiceClient(client_options=client_options)
    model_resource = model_client.get_model(name=tuned_model_name)

    with open(model.path, "w") as f:
        f.write(json_format.MessageToJson(model_resource._pb))

    model.metadata["resourceName"] = tuned_model_name
    logging.info(f"Wrote tuned model details to {model.path}")


@dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        "transformers",
        "datasets",
        "torch",
        "bitsandbytes",
        "accelerate",
        "peft",
    ],
)
def huggingface_optimization_sequence(
    optimization_steps: List[str],
    teacher_model_name: str,
    student_model_name: str,
    huggingface_dataset_name: str,
    optimized_model: dsl.Output[dsl.Model],
) -> None:
    """
    Performs a sequence of optimization steps on a Hugging Face model.
    Steps can include distillation, PEFT tuning, and quantization.
    """
    import logging
    import os
    import shutil
    from datasets import load_dataset
    import torch
    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    logging.basicConfig(level=logging.INFO)

    # The model is always downloaded from the Hub at the start.
    logging.info(f"Downloading initial student model: {student_model_name}")
    current_model_path = os.path.join(optimized_model.path, "download")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    initial_model = AutoModelForMaskedLM.from_pretrained(student_model_name)
    initial_model.save_pretrained(current_model_path)
    tokenizer.save_pretrained(current_model_path)
    logging.info(f"Initial model saved to: {current_model_path}")

    if not optimization_steps:
        logging.info("No optimization steps provided. Passing through original model.")
        # The downloaded model is already in a sub-directory of the output path.
        # We just need to move the contents up one level.
        for item in os.listdir(current_model_path):
            shutil.move(
                os.path.join(current_model_path, item),
                os.path.join(optimized_model.path, item),
            )
        return

    for i, step in enumerate(optimization_steps):
        logging.info(
            f"--- Starting Step {i + 1}/{len(optimization_steps)}: {step} ---"
        )

        if step == "huggingface-distillation":
            tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
            teacher_model_obj = AutoModelForMaskedLM.from_pretrained(
                teacher_model_name
            )
            # For distillation, we start from the original student model, not an intermediate one
            student_model_obj = AutoModelForMaskedLM.from_pretrained(
                student_model_name
            )
            dataset = load_dataset(huggingface_dataset_name, split="train").select(
                range(128)
            )
            tokenized_dataset = dataset.map(
                lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length"),
                batched=True,
            )

            class DistillationTrainer(Trainer):
                def __init__(self, *args, teacher_model=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.teacher = teacher_model
                    self.teacher.to(self.args.device)
                    self.teacher.eval()

                def compute_loss(self, model, inputs, return_outputs=False):
                    s_outputs = model(**inputs)
                    s_loss = s_outputs.loss
                    with torch.no_grad():
                        t_outputs = self.teacher(**inputs)
                    dist_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(s_outputs.logits, dim=-1),
                        torch.nn.functional.softmax(t_outputs.logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss = 0.5 * s_loss + 0.5 * dist_loss
                    return (loss, s_outputs) if return_outputs else loss

            trainer = DistillationTrainer(
                model=student_model_obj,
                teacher_model=teacher_model_obj,
                args=TrainingArguments(
                    output_dir=current_model_path,
                    num_train_epochs=1,
                    per_device_train_batch_size=8,
                    logging_steps=1,
                    report_to=[],
                    overwrite_output_dir=True,
                ),
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            trainer.train()
            trainer.save_model(current_model_path)
            logging.info(f"Distillation complete. Model saved to: {current_model_path}")

        elif step == "huggingface-peft-lora":
            tokenizer = AutoTokenizer.from_pretrained(current_model_path)
            model_to_tune = AutoModelForMaskedLM.from_pretrained(current_model_path)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "key"],
                lora_dropout=0.05,
                bias="none",
            )
            peft_model = get_peft_model(model_to_tune, lora_config)
            peft_model.print_trainable_parameters()
            dataset = load_dataset(huggingface_dataset_name, split="train").select(
                range(128)
            )
            tokenized_dataset = dataset.map(
                lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length"),
                batched=True,
            )
            trainer = Trainer(
                model=peft_model,
                args=TrainingArguments(
                    output_dir=current_model_path,
                    num_train_epochs=1,
                    per_device_train_batch_size=8,
                    logging_steps=10,
                    report_to=[],
                    overwrite_output_dir=True,
                ),
                train_dataset=tokenized_dataset,
            )
            trainer.train()
            trainer.save_model(current_model_path)
            logging.info(f"PEFT tuning complete. Adapters saved to: {current_model_path}")

        elif step == "huggingface-quantization":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(current_model_path)
            quantized_model_obj = AutoModelForMaskedLM.from_pretrained(
                current_model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
            quantized_model_obj.save_pretrained(current_model_path)
            tokenizer.save_pretrained(current_model_path)
            logging.info(f"Quantized model saved to: {current_model_path}")

        else:
            logging.warning(f"Unknown optimization step: {step}. Skipping.")

    logging.info(f"All steps complete. Copying final model to {optimized_model.path}")
    # Clean the output directory before copying the final model artifacts
    for item in os.listdir(optimized_model.path):
        item_path = os.path.join(optimized_model.path, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    for item in os.listdir(current_model_path):
        shutil.move(
            os.path.join(current_model_path, item),
            os.path.join(optimized_model.path, item),
        )


@dsl.component
def prepare_unmanaged_model_artifact(
    model_artifact: dsl.Input[dsl.Model],
    serving_container_image: str,
    unmanaged_model_artifact: dsl.Output[UnmanagedContainerModel],
):
    """Prepares an UnmanagedContainerModel artifact for ModelUploadOp."""
    unmanaged_model_artifact.uri = model_artifact.uri
    unmanaged_model_artifact.metadata["containerSpec"] = {
        "imageUri": serving_container_image
    }


@dsl.component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform>=1.49.0"],
)
def run_text_generation_evaluation(
    project: str,
    location: str,
    model: dsl.Input[VertexModel],
    evaluation_dataset: str,
    target_field_name: str,
    evaluation_display_name: str,
) -> None:
    """
    Launches a Vertex AI evaluation pipeline for text generation models.

    This component uses the google-cloud-aiplatform SDK to run a pre-built
    evaluation pipeline, which is the standard method for evaluating LLMs on
    Vertex AI. The results will be visible in the Google Cloud console.
    """
    import logging

    from google.cloud import aiplatform

    logging.basicConfig(level=logging.INFO)

    model_resource_name = model.metadata["resourceName"]
    logging.info(f"Starting evaluation for model: {model_resource_name}")

    # GCS path to the pre-built evaluation pipeline template
    EVAL_PIPELINE_TEMPLATE_PATH = "https://us-kfp.pkg.dev/ml-pipeline/llm-evaluation/evaluation-llm-text-generation-pipeline/v1.0.0"

    pipeline_job = aiplatform.PipelineJob(
        display_name=evaluation_display_name,
        template_path=EVAL_PIPELINE_TEMPLATE_PATH,
        pipeline_root=model.uri.rsplit("/", 2)[0] + "/evaluation_pipeline_root",
        project=project,
        location=location,
        parameter_values={
            "project": project,
            "location": location,
            "model_name": model_resource_name,
            "dataset_gcs_source": evaluation_dataset,
            "target_field_name": target_field_name,
            "task": "text-generation",
        },
    )
    pipeline_job.run(sync=True)
    logging.info(f"Evaluation pipeline job finished: {pipeline_job.resource_name}")


# ==============================================================================
# MAIN PIPELINE DEFINITION
# ==============================================================================


@dsl.component(
    base_image="python:3.9",
    # packages_to_install=["future-emulator-sdk"], # Example for future use
)
def on_device_emulation(
    model: dsl.Input[VertexModel],
) -> None:
    """
    Placeholder component for on-device emulation.

    This component is intended for future integration with on-device emulators.
    The logic to load the provided model onto an emulated device and run
    performance or accuracy experiments would be added here.
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    model_resource_name = model.metadata["resourceName"]
    logging.info(
        f"--- Placeholder: On-Device Emulation for model {model_resource_name} ---"
    )
    logging.info("This component is a placeholder for future implementation.")
    logging.info("Here you would add code to:")
    logging.info("1. Connect to a device emulator.")
    logging.info("2. Load the model onto the emulated device.")
    logging.info("3. Run experiments (e.g., latency, memory usage, accuracy).")
    logging.info("4. Report results as metrics or output artifacts.")
    pass

@dsl.pipeline(
    name="slm-automotive-optimization-pipeline-sequential",
    description="A sequential, modular pipeline to version, optimize, deploy, and evaluate SLMs.",
)
def slm_automotive_pipeline(
    project: str,
    location: str,
    base_model_display_name: str,
    serving_container_image: str,
    tuning_method: str = "huggingface",
    optimization_steps: List[str] = [
        "huggingface-peft-lora",
        "huggingface-quantization",
    ],
    teacher_model_name: str = "bert-base-uncased",
    student_model_name: str = "prajjwal1/bert-tiny",
    huggingface_dataset_name: str = "imdb",
    deployment_machine_type: str = "n1-standard-4",
    test_dataset_uri: str = "gs://cloud-samples-data/vertex-ai/pipeline_intro/test.csv",
    evaluation_target_field_name: str = "output_text",
    finetuning_dataset_uri: str = "gs://cloud-samples-data/vertex-ai/llm/python/training-data.jsonl",
    base_gemini_model: str = "gemini-1.5-flash-001",
):
    """The main pipeline orchestrating the optimization, deployment, and evaluation of SLMs.

    Args:
        project (str): The Google Cloud project ID.
        location (str): The Google Cloud region for the pipeline.
        base_model_display_name (str): The base display name for models created by this pipeline.
        serving_container_image (str): The URI of the container image for serving the model.
        tuning_method (str): The optimization method to use ('gemini-supervised-finetuning' or 'huggingface').
        optimization_steps (List[str]): A list of optimization steps for the Hugging Face branch.
        teacher_model_name (str): The Hugging Face model name for the teacher model in distillation.
        student_model_name (str): The Hugging Face model name for the student model.
        huggingface_dataset_name (str): The name of the dataset on Hugging Face Hub to use for training.
        deployment_machine_type (str): The machine type for the deployed model endpoint.
        test_dataset_uri (str): The GCS URI of the dataset for final model evaluation.
        evaluation_target_field_name (str): The name of the target field in the evaluation dataset.
        finetuning_dataset_uri (str): The GCS URI of the dataset for Gemini fine-tuning.
        base_gemini_model (str): The base Gemini model to use for fine-tuning.
    """
    pipeline_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    versioned_model_name = f"{base_model_display_name}-{pipeline_timestamp}"

    # Step 2: Conditional Optimization.
    with dsl.If(tuning_method == "gemini-supervised-finetuning", name="gemini-branch"):
        gemini_op = gemini_supervised_finetuning(
            project=project,
            location=location,
            model_display_name=versioned_model_name,
            finetuning_dataset_uri=finetuning_dataset_uri,
            base_gemini_model=base_gemini_model,
        )
    with dsl.Else(name="huggingface-branch"):
        # The Hugging Face component produces a file-based model artifact.
        hf_op = huggingface_optimization_sequence(
            optimization_steps=optimization_steps,
            teacher_model_name=teacher_model_name,
            student_model_name=student_model_name,
            huggingface_dataset_name=huggingface_dataset_name,
        )
        # We then prepare an UnmanagedContainerModel artifact from the output
        # of the HF sequence, adding the necessary container spec metadata.
        prepare_model_op = prepare_unmanaged_model_artifact(
            model_artifact=hf_op.outputs["optimized_model"],
            serving_container_image=serving_container_image,
        ).set_display_name("Prepare HF Model Artifact")

        # Finally, upload the prepared artifact to create a Vertex AI Model.
        hf_upload_op = ModelUploadOp(
            project=project,
            location=location,
            display_name=versioned_model_name,
            unmanaged_container_model=prepare_model_op.outputs[
                "unmanaged_model_artifact"
            ],
        ).set_display_name("Upload Optimized HF Model")

    # dsl.OneOf merges the model outputs from the different conditional branches.
    # Both branches now produce a compatible google.VertexModel artifact named 'model'.
    optimized_model = dsl.OneOf(
        gemini_op.outputs["model"],
        hf_upload_op.outputs["model"],
    )

    # Step 3: Evaluation
    eval_op = run_text_generation_evaluation(
        project=project,
        location=location,
        model=optimized_model,
        evaluation_dataset=test_dataset_uri,
        target_field_name=evaluation_target_field_name,
        evaluation_display_name=f"eval-{versioned_model_name}",
    )

    # Step 4: On-Device Emulation (runs after evaluation)
    emulation_op = on_device_emulation(model=optimized_model)
    emulation_op.after(eval_op)

    # Step 5: Deployment (runs after evaluation and emulation are complete)
    endpoint_create_op = EndpointCreateOp(
        project=project,
        display_name=f"slm-endpoint-{versioned_model_name}",
    )

    deploy_op = ModelDeployOp(
        endpoint=endpoint_create_op.outputs["endpoint"],
        model=optimized_model,
        deployed_model_display_name=versioned_model_name,
        dedicated_resources_machine_type=deployment_machine_type,
        traffic_split={"0": 100},
    )
    deploy_op.after(emulation_op)


if __name__ == "__main__":
    compiled_dir = "compiled"
    if not os.path.exists(compiled_dir):
        os.makedirs(compiled_dir)
    pipeline_json_path = os.path.join(
        compiled_dir, "slm_automotive_pipeline_sequential.json"
    )
    compiler.Compiler().compile(
        pipeline_func=slm_automotive_pipeline,
        package_path=pipeline_json_path,
    )
    print(f"Pipeline compiled successfully to: {pipeline_json_path}")

    # --------------------------------------------------------------------------
    # Optional: Submit the pipeline to Vertex AI
    # --------------------------------------------------------------------------
    SUBMIT_TO_VERTEX = True  # Set to True to submit the pipeline

    if SUBMIT_TO_VERTEX:
        import google.auth
        from google.api_core import exceptions
        from google.cloud import aiplatform, storage

        # Discover project ID from credentials.
        try:
            credentials, PROJECT_ID = google.auth.default()
            print(f"Discovered project ID: {PROJECT_ID}")
        except google.auth.exceptions.DefaultCredentialsError:
            print(
                "Could not automatically discover project ID. "
                "Please run 'gcloud auth application-default login' or set the GOOGLE_CLOUD_PROJECT environment variable."
            )
            raise

        # --- PLEASE CONFIGURE THESE VALUES ---
        LOCATION = "europe-west4"  # The region for the pipeline job
        # A sensible default for the pipeline root. You may need to create this bucket.
        PIPELINE_ROOT = (
            f"gs://{PROJECT_ID}-vertex-pipelines/slm-optimization-root"
        )

        # Create the GCS bucket for the pipeline root if it doesn't exist.
        bucket_name = PIPELINE_ROOT.split("/")[2]
        storage_client = storage.Client(project=PROJECT_ID)
        try:
            storage_client.get_bucket(bucket_name)
            print(f"Bucket {bucket_name} already exists.")
        except exceptions.NotFound:
            print(f"Bucket {bucket_name} not found. Creating bucket...")
            storage_client.create_bucket(bucket_name, location=LOCATION)
            print(f"Bucket {bucket_name} created successfully in {LOCATION}.")
        # ---

        # Required pipeline parameters
        BASE_MODEL_DISPLAY_NAME = "my-slm-model"
        SERVING_CONTAINER_IMAGE = "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-12:latest"  # Example
        # ---

        if "your-gcp-region" in LOCATION:
            raise ValueError("Please configure the LOCATION for the pipeline run.")

        print("\nSubmitting pipeline to Vertex AI...")
        aiplatform.init(project=PROJECT_ID, location=LOCATION)

        pipeline_params = {
            "project": PROJECT_ID,
            "location": LOCATION,
            "base_model_display_name": BASE_MODEL_DISPLAY_NAME,
            "serving_container_image": SERVING_CONTAINER_IMAGE,
        }

        job = aiplatform.PipelineJob(
            display_name="slm-optimization-pipeline",
            template_path=pipeline_json_path,
            pipeline_root=PIPELINE_ROOT,
            parameter_values=pipeline_params,
            enable_caching=True,
        )

        job.submit()
        print(f"Pipeline job submitted. View it in the console: {job._dashboard_uri}")
