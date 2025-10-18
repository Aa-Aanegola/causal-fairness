#!/bin/bash
#SBATCH --job-name=celeba_array
#SBATCH --output=logs/celeba_array_%A_%a.out
#SBATCH --error=logs/celeba_array_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --account=edu
#SBATCH --array=1-3

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
echo "Starting CelebA training job array at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"

# Load modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision pytorch-lightning lightly torchmetrics pyyaml pillow numpy tqdm

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create checkpoint directories
mkdir -p ckpt/celeba/encoder
mkdir -p ckpt/celeba/oracle  
mkdir -p ckpt/celeba/student

# Define training tasks
case $SLURM_ARRAY_TASK_ID in
    1)
        echo "=========================================="
        echo "TASK 1: SimCLR Training"
        echo "=========================================="
        python train_celeba_simclr.py
        ;;
    2)
        echo "=========================================="
        echo "TASK 2: Teacher Training"
        echo "=========================================="
        python train_celeba_teacher.py
        ;;
    3)
        echo "=========================================="
        echo "TASK 3: Student Training"
        echo "=========================================="
        # Wait for dependencies
        echo "Waiting for SimCLR and Teacher checkpoints..."
        while [ ! -f "ckpt/celeba/encoder/simclr-celeba.ckpt" ] || [ ! -f "ckpt/celeba/oracle/teacher-celeba.ckpt" ]; do
            echo "Waiting for dependencies..."
            sleep 60
        done
        echo "Dependencies found, starting student training..."
        python train_celeba_student.py
        ;;
    *)
        echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID completed at $(date)"
