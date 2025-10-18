#!/usr/bin/env python3
"""
Test script for CelebA causal fairness pipeline.
Tests the complete pipeline with a small subset of data.
"""

import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import os

from celeba_dataset import CelebACausalDataset, CelebASimCLRDataset
from celeba_utils import create_celeba_causal_data, CelebADatasetWithCovars
from model import SimCLRModel, TeacherModel, StudentTrainer

def test_simclr_training():
    """Test SimCLR training on a small subset."""
    print("Testing SimCLR training...")
    
    # Load config
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create small dataset for testing
    root_dir = cfg["data"]["root_dir"]
    dataset = CelebASimCLRDataset(root_dir=root_dir, split='train')
    
    # Use only first 1000 samples for testing
    subset = Subset(dataset, range(min(1000, len(dataset))))
    
    dataloader = DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # Create model
    model = SimCLRModel(cfg["simclr"])
    
    # Create trainer with minimal epochs
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",  # Use CPU for testing
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )
    
    print(f"Training on {len(subset)} samples for 2 epochs...")
    trainer.fit(model, dataloader)
    print("SimCLR training test completed!")

def test_teacher_training():
    """Test teacher training on a small subset."""
    print("Testing teacher training...")
    
    # Load config
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Create small dataset for testing
    root_dir = cfg["data"]["root_dir"]
    data = create_celeba_causal_data(root_dir=root_dir, split='train')
    
    # Use only first 1000 samples
    subset_size = min(1000, len(data['image']))
    subset_indices = list(range(subset_size))
    
    dataset = CelebADatasetWithCovars(data, subset_indices)
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # Create model
    model = TeacherModel(cfg["oracle"])
    
    # Create trainer with minimal epochs
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",  # Use CPU for testing
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False
    )
    
    print(f"Training on {len(dataset)} samples for 2 epochs...")
    trainer.fit(model, dataloader)
    print("Teacher training test completed!")

def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    
    # Load config
    with open("celeba_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    root_dir = cfg["data"]["root_dir"]
    
    # Test causal dataset
    dataset = CelebACausalDataset(root_dir=root_dir, split='train')
    print(f"Causal dataset size: {len(dataset)}")
    
    # Test a few samples
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: {[(k, v.shape if hasattr(v, 'shape') else v) for k, v in sample.items()]}")
    
    # Test SimCLR dataset
    simclr_dataset = CelebASimCLRDataset(root_dir=root_dir, split='train')
    print(f"SimCLR dataset size: {len(simclr_dataset)}")
    
    # Test a sample
    sample = simclr_dataset[0]
    print(f"SimCLR sample: {[(k, v.shape if hasattr(v, 'shape') else v) for k, v in zip(['views', 'target1', 'target2'], sample)]}")
    
    print("Data loading test completed!")

def main():
    """Run all tests."""
    print("Starting CelebA pipeline tests...")
    
    # Test data loading first
    test_data_loading()
    print()
    
    # Test SimCLR training
    test_simclr_training()
    print()
    
    # Test teacher training
    test_teacher_training()
    print()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
