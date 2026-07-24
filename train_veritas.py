"""
Training script for VERITAS model.

This script sets up and runs the complete training pipeline for VERITAS.

Usage:
    python train_veritas.py
    
Or in Google Colab:
    !python train_veritas.py
"""

import torch
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import Config
from src.models import VERITASModel, MultiTaskLoss
from src.data import create_dataloaders
from src.training import Trainer, TrainingConfig, log_experiment
from src.utils.reproducibility import set_random_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    
    print("\n" + "="*70)
    print("VERITAS Training Pipeline")
    print("="*70 + "\n")
    
    # =========================================================================
    # 1. Configuration
    # =========================================================================
    
    logger.info("Loading configuration...")
    
    # Model configuration
    model_config = Config({
        'model': {
            'input_resolution': 600,
            'use_multi_representation': True,  # Set to False for RGB-only baseline
            'use_full_segmentation': True
        },
        'representations': {
            'imagenet_mean': [0.485, 0.456, 0.406],
            'imagenet_std': [0.229, 0.224, 0.225],
            'srm_filter_count': 3,
            'noise_normalization': 'tanh',
            'dct_block_size': 8,
            'frequency_bands': ['low', 'mid', 'high']
        }
    })
    
    # Training configuration
    training_config = TrainingConfig(
        # Optimizer
        learning_rate=1e-4,
        weight_decay=0.01,
        optimizer_name='adamw',
        
        # Training
        num_epochs=50,
        batch_size=4,  # Adjust based on GPU memory
        gradient_clip_max_norm=1.0,
        
        # Scheduler
        use_scheduler=True,
        scheduler_name='reduce_on_plateau',
        scheduler_patience=5,
        scheduler_factor=0.5,
        scheduler_min_lr=1e-7,
        
        # Checkpointing (Google Drive)
        checkpoint_dir='/content/drive/MyDrive/VERITAS/checkpoints',
        save_every_n_epochs=5,
        save_best_only=False,  # Keep regular checkpoints too
        best_metric='val_total_loss',
        best_metric_mode='min',
        
        # Logging
        log_every_n_steps=10,
        log_dir='/content/drive/MyDrive/VERITAS/logs',
        
        # Reproducibility
        random_seed=42,
        
        # Early stopping
        use_early_stopping=True,
        early_stopping_patience=10
    )
    
    logger.info(f"Model config: use_multi_representation={model_config.get('model.use_multi_representation')}")
    logger.info(f"Training config: lr={training_config.learning_rate}, epochs={training_config.num_epochs}")
    
    # =========================================================================
    # 2. Set Random Seeds for Reproducibility
    # =========================================================================
    
    logger.info("Setting random seeds...")
    seeds = set_random_seed(training_config.random_seed)
    
    # =========================================================================
    # 3. Create Data Loaders
    # =========================================================================
    
    logger.info("Creating data loaders...")
    
    # Dataset paths (adjust for your setup)
    dataset_path = '/content/drive/MyDrive/VERITAS/dataset'
    
    # Create data loaders
    # NOTE: You need to implement create_dataloaders in src/data/__init__.py
    # For now, this is a placeholder showing the expected interface
    try:
        from src.data import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_path=dataset_path,
            batch_size=training_config.batch_size,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
    except ImportError:
        logger.error("create_dataloaders not yet implemented!")
        logger.info("You need to implement create_dataloaders in src/data/__init__.py")
        logger.info("For now, you can use the existing VERITASDataset class")
        
        # Example fallback (you'll need to adapt this):
        from src.data.dataset import VERITASDataset
        from torch.utils.data import DataLoader
        
        # Load your dataset (adjust paths)
        train_dataset = VERITASDataset(
            annotation_file=f'{dataset_path}/Train/Train_poly.json',
            image_dir=f'{dataset_path}/Train',
            config=model_config,
            transform=None  # Add transforms if needed
        )
        
        val_dataset = VERITASDataset(
            annotation_file=f'{dataset_path}/Val/Val_poly.json',
            image_dir=f'{dataset_path}/Val',
            config=model_config,
            transform=None
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = None  # Test set not used during training
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
    
    # =========================================================================
    # 4. Create Model
    # =========================================================================
    
    logger.info("Creating VERITAS model...")
    
    model = VERITASModel(
        config=model_config,
        use_multi_representation=model_config.get('model.use_multi_representation'),
        use_full_segmentation=model_config.get('model.use_full_segmentation')
    )
    
    # Log model info
    num_params = model.get_num_parameters()
    model_size = model.get_model_size_mb()
    arch_summary = model.get_architecture_summary()
    
    logger.info(f"Model: {arch_summary['backbone']}")
    logger.info(f"Parameters: {num_params:,}")
    logger.info(f"Size: {model_size:.2f} MB")
    logger.info(f"Multi-representation: {arch_summary['multi_representation']}")
    logger.info(f"Full segmentation: {arch_summary['full_segmentation']}")
    
    # =========================================================================
    # 5. Create Loss Function
    # =========================================================================
    
    logger.info("Creating multi-task loss...")
    
    loss_fn = MultiTaskLoss(
        cls_weight=0.4,  # Classification loss weight
        seg_weight=0.6   # Segmentation loss weight
    )
    
    # =========================================================================
    # 6. Log Experiment Information
    # =========================================================================
    
    logger.info("Logging reproducibility information...")
    
    dataset_info = {
        'dataset_name': 'OpenForensics',
        'dataset_path': dataset_path,
        'split_info': {
            'train_samples': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'N/A',
            'val_samples': len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'N/A',
            'batch_size': training_config.batch_size
        }
    }
    
    repro_logger = log_experiment(
        log_dir=training_config.log_dir,
        model=model,
        config=training_config.to_dict(),
        seeds=seeds,
        dataset_info=dataset_info
    )
    
    # =========================================================================
    # 7. Create Trainer
    # =========================================================================
    
    logger.info("Creating trainer...")
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # =========================================================================
    # 8. Train Model
    # =========================================================================
    
    logger.info("\nStarting training...\n")
    
    try:
        # Train
        trainer.fit()
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_path = Path(training_config.log_dir) / "training_history.png"
        trainer.plot_history(save_path=str(plot_path))
        
        logger.info(f"\n{'='*70}")
        logger.info("Training completed successfully!")
        logger.info(f"{'='*70}\n")
        
        logger.info(f"Best {training_config.best_metric}: {trainer.best_metric_value:.4f}")
        logger.info(f"Checkpoints saved to: {training_config.checkpoint_dir}")
        logger.info(f"Logs saved to: {training_config.log_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        logger.info("Saving current checkpoint...")
        
        # Save checkpoint on interrupt
        current_metrics = {
            'train_loss': trainer.history['train_loss'][-1] if trainer.history['train_loss'] else 0,
            'val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else 0
        }
        trainer.save_checkpoint(current_metrics, is_best=False, filename="interrupted_checkpoint.pth")
        
        logger.info("Checkpoint saved. You can resume training later.")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
