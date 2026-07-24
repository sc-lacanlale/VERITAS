"""
Test training pipeline components.

Quick validation that all training components work correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.models import VERITASModel, MultiTaskLoss
from src.training import Trainer, TrainingConfig, CheckpointManager, log_experiment
from src.utils.reproducibility import set_random_seed


def test_training_config():
    """Test training configuration."""
    print("\n" + "="*70)
    print("Test 1: Training Configuration")
    print("="*70)
    
    config = TrainingConfig(
        learning_rate=1e-4,
        num_epochs=2,
        batch_size=2
    )
    
    print(f"✓ Created training config")
    print(f"  - LR: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    
    # Test save/load
    config.save('/tmp/test_config.json')
    loaded_config = TrainingConfig.load('/tmp/test_config.json')
    assert loaded_config.learning_rate == config.learning_rate
    print(f"✓ Config save/load works")


def test_checkpoint_manager():
    """Test checkpoint manager."""
    print("\n" + "="*70)
    print("Test 2: Checkpoint Manager")
    print("="*70)
    
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir='/tmp/test_checkpoints',
        keep_last_n=3,
        keep_best_n=2
    )
    
    print(f"✓ Created checkpoint manager")
    print(f"  - Directory: /tmp/test_checkpoints")
    print(f"  - Keep last: 3")
    print(f"  - Keep best: 2")
    
    # Test with dummy model
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Save checkpoint
    checkpoint_path = checkpoint_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        metrics={'val_loss': 0.5},
        is_best=True
    )
    
    print(f"✓ Saved checkpoint: {Path(checkpoint_path).name}")
    
    # Load checkpoint
    checkpoint = checkpoint_mgr.load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        device='cpu'
    )
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")


def test_reproducibility_logger():
    """Test reproducibility logger."""
    print("\n" + "="*70)
    print("Test 3: Reproducibility Logger")
    print("="*70)
    
    # Set seeds
    seeds = set_random_seed(42)
    print(f"✓ Set random seeds: {seeds}")
    
    # Create model
    config = Config({'model': {'input_resolution': 600}})
    model = VERITASModel(config, use_multi_representation=False)
    
    # Log experiment
    repro_logger = log_experiment(
        log_dir='/tmp/test_logs',
        model=model,
        config={'learning_rate': 1e-4, 'batch_size': 4},
        seeds=seeds,
        dataset_info={
            'dataset_name': 'Test Dataset',
            'dataset_path': '/tmp/data',
            'split_info': {'train': 1000, 'val': 200}
        }
    )
    
    print(f"✓ Logged experiment information")
    print(f"  - Environment: {len(repro_logger.info['environment'])} items")
    print(f"  - Seeds: {list(repro_logger.info['seeds'].keys())}")
    print(f"  - Model: {repro_logger.info['model']['num_parameters']:,} params")


def test_trainer_initialization():
    """Test trainer initialization."""
    print("\n" + "="*70)
    print("Test 4: Trainer Initialization")
    print("="*70)
    
    # Create dummy data loaders
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 600, 600),
                'mask': torch.randint(0, 2, (1, 600, 600)).float(),
                'label': torch.tensor([float(idx % 2)])
            }
    
    train_loader = torch.utils.data.DataLoader(
        DummyDataset(),
        batch_size=2,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        DummyDataset(),
        batch_size=2,
        shuffle=False
    )
    
    print(f"✓ Created dummy data loaders")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    
    # Create model and loss
    config = Config({'model': {'input_resolution': 600}})
    model = VERITASModel(config, use_multi_representation=False)
    loss_fn = MultiTaskLoss()
    
    print(f"✓ Created model and loss")
    
    # Create trainer
    training_config = TrainingConfig(
        num_epochs=2,
        batch_size=2,
        checkpoint_dir='/tmp/test_checkpoints',
        log_dir='/tmp/test_logs',
        device='cpu'  # Use CPU for testing
    )
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    print(f"✓ Created trainer")
    print(f"  - Device: {trainer.device}")
    print(f"  - Optimizer: {training_config.optimizer_name}")
    print(f"  - Scheduler: {training_config.scheduler_name if training_config.use_scheduler else 'None'}")


def test_single_training_step():
    """Test single training step."""
    print("\n" + "="*70)
    print("Test 5: Single Training Step")
    print("="*70)
    
    # Create dummy data
    batch = {
        'image': torch.randn(2, 3, 600, 600),
        'mask': torch.randint(0, 2, (2, 1, 600, 600)).float(),
        'label': torch.tensor([[0.0], [1.0]])
    }
    
    # Create model and loss
    config = Config({'model': {'input_resolution': 600}})
    model = VERITASModel(config, use_multi_representation=False)
    loss_fn = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    outputs = model(batch['image'])
    print(f"✓ Forward pass successful")
    print(f"  - Classification logit: {outputs['classification_logit'].shape}")
    print(f"  - Segmentation logits: {outputs['segmentation_logits'].shape}")
    
    # Compute loss
    loss_dict = loss_fn(
        predictions=outputs,
        targets={'label': batch['label'], 'mask': batch['mask']}
    )
    
    print(f"✓ Loss computation successful")
    print(f"  - Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - Classification loss: {loss_dict['classification_loss'].item():.4f}")
    print(f"  - Segmentation loss: {loss_dict['segmentation_loss'].item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss_dict['total_loss'].backward()
    optimizer.step()
    
    print(f"✓ Backward pass successful")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Testing VERITAS Training Pipeline")
    print("="*70)
    
    try:
        test_training_config()
        test_checkpoint_manager()
        test_reproducibility_logger()
        test_trainer_initialization()
        test_single_training_step()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\nTraining pipeline is ready to use!")
        print("Next steps:")
        print("  1. Prepare your dataset")
        print("  2. Run: python train_veritas.py")
        print("  3. Monitor training progress")
        print("\nSee TRAINING_PIPELINE_COMPLETE.md for full documentation.")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
