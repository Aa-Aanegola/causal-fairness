#!/bin/bash
#SBATCH --job-name=celeba_pipeline
#SBATCH --output=logs/celeba_pipeline_%j.out
#SBATCH --error=logs/celeba_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --account=edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
echo "Starting CelebA causal fairness pipeline at $(date)"
echo "Job ID: $SLURM_JOB_ID"
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

# Function to check if checkpoint exists
check_checkpoint() {
    local checkpoint_path="$1"
    if [ -f "$checkpoint_path" ]; then
        echo "✓ Checkpoint found: $checkpoint_path"
        return 0
    else
        echo "✗ Checkpoint missing: $checkpoint_path"
        return 1
    fi
}

# Function to run training with error handling
run_training() {
    local script_name="$1"
    local job_name="$2"
    local checkpoint_path="$3"
    
    echo "=========================================="
    echo "Starting $job_name training..."
    echo "Script: $script_name"
    echo "Checkpoint path: $checkpoint_path"
    echo "=========================================="
    
    # Check if checkpoint already exists
    if [ -n "$checkpoint_path" ] && check_checkpoint "$checkpoint_path"; then
        echo "Checkpoint already exists, skipping $job_name training"
        return 0
    fi
    
    # Run the training script
    if python "$script_name"; then
        echo "✓ $job_name training completed successfully"
        
        # Verify checkpoint was created
        if [ -n "$checkpoint_path" ] && check_checkpoint "$checkpoint_path"; then
            echo "✓ Checkpoint verified: $checkpoint_path"
        else
            echo "⚠ Warning: Expected checkpoint not found: $checkpoint_path"
        fi
        return 0
    else
        echo "✗ $job_name training failed"
        return 1
    fi
}

# Create checkpoint directories
mkdir -p ckpt/celeba/encoder
mkdir -p ckpt/celeba/oracle  
mkdir -p ckpt/celeba/student

# Step 1: Run SimCLR training (self-supervised representation learning)
echo "=========================================="
echo "STEP 1: SimCLR Training"
echo "=========================================="
SIMCLR_CHECKPOINT="ckpt/celeba/encoder/simclr-celeba.ckpt"
if ! run_training "train_celeba_simclr.py" "SimCLR" "$SIMCLR_CHECKPOINT"; then
    echo "SimCLR training failed, exiting pipeline"
    exit 1
fi

# Step 2: Run Teacher training (oracle model) - can run in parallel with SimCLR
echo "=========================================="
echo "STEP 2: Teacher Training"
echo "=========================================="
TEACHER_CHECKPOINT="ckpt/celeba/oracle/teacher-celeba.ckpt"
if ! run_training "train_celeba_teacher.py" "Teacher" "$TEACHER_CHECKPOINT"; then
    echo "Teacher training failed, exiting pipeline"
    exit 1
fi

# Step 3: Run Student training (knowledge distillation) - depends on both SimCLR and Teacher
echo "=========================================="
echo "STEP 3: Student Training"
echo "=========================================="
STUDENT_CHECKPOINT="ckpt/celeba/student/student-celeba.ckpt"

# Verify dependencies exist
if ! check_checkpoint "$SIMCLR_CHECKPOINT"; then
    echo "SimCLR checkpoint missing, cannot proceed with student training"
    exit 1
fi

if ! check_checkpoint "$TEACHER_CHECKPOINT"; then
    echo "Teacher checkpoint missing, cannot proceed with student training"
    exit 1
fi

if ! run_training "train_celeba_student.py" "Student" "$STUDENT_CHECKPOINT"; then
    echo "Student training failed"
    exit 1
fi

# Step 4: Run pipeline test
echo "=========================================="
echo "STEP 4: Pipeline Testing"
echo "=========================================="
echo "Running pipeline test..."
if python test_celeba_pipeline.py; then
    echo "✓ Pipeline test completed successfully"
else
    echo "⚠ Pipeline test failed, but training completed"
fi

# Final summary
echo "=========================================="
echo "PIPELINE COMPLETION SUMMARY"
echo "=========================================="
echo "Completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Check final checkpoints
echo "Final checkpoint status:"
check_checkpoint "$SIMCLR_CHECKPOINT"
check_checkpoint "$TEACHER_CHECKPOINT" 
check_checkpoint "$STUDENT_CHECKPOINT"

# List output files
echo "Generated files:"
ls -la ckpt/celeba/encoder/
ls -la ckpt/celeba/oracle/
ls -la ckpt/celeba/student/

echo "CelebA causal fairness pipeline completed!"
