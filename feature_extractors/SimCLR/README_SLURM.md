# CelebA Causal Fairness Training Pipeline

This directory contains SLURM scripts to run the complete CelebA causal fairness training pipeline.

## Overview

The pipeline consists of three main training stages:

1. **SimCLR Training** (`train_celeba_simclr.py`): Self-supervised representation learning
   - Trains for 200 epochs
   - Saves encoder checkpoint to `ckpt/celeba/encoder/`

2. **Teacher Training** (`train_celeba_teacher.py`): Oracle model training
   - Trains for 30 epochs  
   - Saves teacher checkpoint to `ckpt/celeba/oracle/`

3. **Student Training** (`train_celeba_student.py`): Knowledge distillation
   - Trains for 10 epochs
   - Depends on both SimCLR and Teacher checkpoints
   - Saves student checkpoint to `ckpt/celeba/student/`

## Quick Start

### Option 1: Sequential Pipeline (Recommended)
```bash
./submit_celeba_jobs.sh sequential
```

### Option 2: Job Array (Parallel)
```bash
./submit_celeba_jobs.sh array
```

## Manual Submission

### Sequential Pipeline
```bash
sbatch run_celeba_pipeline.sbatch
```

### Job Array
```bash
sbatch run_celeba_array.sbatch
```

## Configuration

Edit `celeba_config.yaml` to adjust:
- Data paths
- Training parameters
- Model architecture
- Resource requirements

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/celeba_pipeline_<JOB_ID>.out

# Cancel job
scancel <JOB_ID>
```

## Resource Requirements

- **GPU**: 1 GPU per job
- **Memory**: 32GB RAM
- **CPU**: 8 cores
- **Time**: 12-24 hours total
- **Storage**: ~10GB for checkpoints and logs

## Output Files

After successful completion:
- `ckpt/celeba/encoder/simclr-celeba.ckpt` - SimCLR encoder
- `ckpt/celeba/oracle/teacher-celeba.ckpt` - Teacher model
- `ckpt/celeba/student/student-celeba.ckpt` - Student model
- `ckpt/celeba/student/celeba_data_with_embeddings.pt` - Extracted embeddings

## Troubleshooting

1. **Missing dependencies**: The script automatically installs required packages
2. **Data path issues**: Update `root_dir` in `celeba_config.yaml`
3. **GPU memory errors**: Reduce `batch_size` in config
4. **Checkpoint errors**: Check file permissions and disk space

## Testing

Run the test pipeline:
```bash
python test_celeba_pipeline.py
```
