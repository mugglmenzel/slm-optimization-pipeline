# SLM Optimization Pipeline for Vertex AI

This repository contains a comprehensive Kubeflow pipeline designed to build, optimize, and evaluate Small Language Models (SLMs) on Google Cloud's Vertex AI platform. It provides a modular and configurable workflow for experimenting with different optimization techniques for automotive and other edge-computing use cases.

## Overview

The core of this project is a sequential Kubeflow pipeline that orchestrates the following high-level steps:
1.  **Model Optimization**: Choose between two powerful optimization paths:
    *   **Gemini Supervised Fine-Tuning**: Fine-tune a base Google Gemini model on your custom dataset.
    *   **Hugging Face Optimization Sequence**: Apply a series of optimizations to a Hugging Face model, such as distillation, PEFT/LoRA tuning, and quantization.
2.  **Evaluation**: Automatically launch a Vertex AI Evaluation job to assess the performance of the optimized model on a test dataset.
3.  **Emulation (Placeholder)**: A dedicated step to integrate on-device emulators for performance testing (e.g., latency, memory usage).
4.  **Deployment**: Deploy the final, optimized model to a Vertex AI Endpoint for real-time inference.

The pipeline is designed to be flexible, allowing users to easily configure models, datasets, and optimization strategies through pipeline parameters.

## Features

-   **Dual Optimization Paths**: Supports both Google's Gemini models and open-source Hugging Face models.
-   **Modular Components**: Each step in the pipeline is a self-contained, reusable component.
-   **Conditional Logic**: The pipeline uses conditional logic (`dsl.If`/`dsl.Else`) to dynamically select the optimization path at runtime.
-   **Automated Deployment**: Seamlessly deploys the best model to a Vertex AI Endpoint.
-   **Automated Evaluation**: Kicks off a standard Vertex AI evaluation job for text generation tasks.
-   **Modern Packaging**: Uses `poetry` for dependency management and execution.
-   **Automatic Resource Creation**: The submission script can automatically create the necessary Google Cloud Storage bucket for pipeline artifacts.

## Prerequisites

1.  **Google Cloud SDK**: Ensure you have `gcloud` installed and configured.
2.  **Authentication**: Authenticate with Google Cloud for application-default credentials:
    ```bash
    gcloud auth application-default login
    ```
3.  **Python 3.9+**
4.  **Poetry**: Follow the installation instructions to install Poetry.

## How to Run

The pipeline can be compiled and submitted to Vertex AI using a few simple steps.

### 1. Setup the Environment

Install the project dependencies using Poetry. This will create a virtual environment and install all packages listed in `pyproject.toml`.

```bash
poetry install
```

### 2. Configure the Pipeline Run

Before submitting, open `pipeline.py` and configure the submission block at the bottom of the file:

-   Set `SUBMIT_TO_VERTEX = True`.
-   Set the `LOCATION` variable to your desired Google Cloud region (e.g., `europe-west4`).
-   (Optional) Adjust other pipeline parameters like `BASE_MODEL_DISPLAY_NAME` or `SERVING_CONTAINER_IMAGE`.

The script will automatically discover your `PROJECT_ID` and create a default `PIPELINE_ROOT` GCS bucket if it doesn't exist.

### 3. Compile and Submit

Run the pipeline.py script using poetry run. This command executes the script within the Poetry-managed virtual environment. The script will first compile the pipeline to a JSON file and then, if configured, submit it as a new job to Vertex AI Pipelines.

```bash
poetry run python pipeline.py
```

You will see a link to the pipeline run in your terminal output.
