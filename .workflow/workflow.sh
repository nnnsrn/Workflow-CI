#!/bin/bash

echo "=== Starting CI Workflow ==="

echo "1. Creating conda environment..."
conda env create -f MLProject/conda.yaml

echo "2. Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mlflow-env

echo "3. Installing MLflow..."
pip install mlflow

echo "4. Running MLflow project..."
mlflow run MLProject

echo "=== CI Workflow Finished ==="
