#!/bin/bash
# CelebA Pipeline Submission Script
# Usage: ./submit_celeba_jobs.sh [sequential|array]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

echo "CelebA Causal Fairness Pipeline Submission"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"

# Check if config file exists
if [ ! -f "celeba_config.yaml" ]; then
    echo "Error: celeba_config.yaml not found!"
    exit 1
fi

# Check if data directory exists
DATA_DIR=$(grep "root_dir:" celeba_config.yaml | awk '{print $2}')
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please update celeba_config.yaml with the correct data path"
    exit 1
fi

echo "Data directory: $DATA_DIR"

# Choose submission method
METHOD=${1:-sequential}

case $METHOD in
    sequential)
        echo "Submitting sequential pipeline job..."
        JOB_ID=$(sbatch run_celeba_pipeline.sbatch | awk '{print $4}')
        echo "Sequential pipeline submitted with Job ID: $JOB_ID"
        echo "Monitor with: squeue -j $JOB_ID"
        echo "View logs: tail -f logs/celeba_pipeline_${JOB_ID}.out"
        ;;
    array)
        echo "Submitting job array..."
        JOB_ID=$(sbatch run_celeba_array.sbatch | awk '{print $4}')
        echo "Job array submitted with Job ID: $JOB_ID"
        echo "Monitor with: squeue -j $JOB_ID"
        echo "View logs: tail -f logs/celeba_array_${JOB_ID}_*.out"
        ;;
    *)
        echo "Usage: $0 [sequential|array]"
        echo "  sequential: Run all jobs in sequence (recommended for first run)"
        echo "  array: Run jobs in parallel array (faster but requires dependency management)"
        exit 1
        ;;
esac

echo ""
echo "Job submission complete!"
echo "Check job status with: squeue -u \$USER"
echo "Cancel job with: scancel $JOB_ID"
