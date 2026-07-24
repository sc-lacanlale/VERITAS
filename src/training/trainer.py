"""
Training pipeline for VERITAS model.

Implements complete training loop with:
- Training and validation epochs
- Checkpoint management (Google Drive)
- Learning rate scheduling
- Gradient clipping
- Reproducibility logging
- Progress monitoring

Requirements: 13, 21
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    optimizer_name: str = 'adamw'
    
    # Training settings
    num_epochs: int = 50
    batch_size: int = 4
    gradient_clip_max_norm: float = 1.0
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_name: str = 'reduce_on_plateau'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Checkpoint settings
    checkpoint_dir: str = '/content/drive/MyDrive/VERITAS/checkpoints'
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    best_metric: str = 'val_total_loss'  # 'val_total_loss', 'val_accuracy', etc.
    best_metric_mode: str = 'min'  # 'min' or 'max'
    
    # Logging
    log_every_n_steps: int = 10
    log_dir: str = '/content/drive/MyDrive/VERITAS/logs'
    
    # Reproducibility
    random_seed: int = 42
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class Trainer:
    """
    VERITAS model trainer with checkpoint management and monitoring.
    
    Handles:
    - Training loop with forward/backward pass
    - Validation loop
    - Checkpoint saving/loading (Google Drive)
    - Learning rate scheduling
    - Gradient clipping
    - Progress logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: VERITAS model
            loss_fn: Multi-task loss function
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader (for final evaluation only)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler() if config.use_scheduler else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric_value = float('inf') if config.best_metric_mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_seg_loss': [],
            'val_seg_loss': [],
            'learning_rates': []
        }
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Optimizer: {config.optimizer_name}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Num epochs: {config.num_epochs}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer_name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
        elif self.config.optimizer_name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler_name.lower() == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.best_metric_mode,
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor,
                min_lr=self.config.scheduler_min_lr,
                verbose=True
            )
        elif self.config.scheduler_name.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.scheduler_min_lr
            )
        elif self.config.scheduler_name.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            logger.warning(f"Unknown scheduler: {self.config.scheduler_name}, using no scheduler")
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_seg_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs,
                targets={
                    'label': labels,
                    'mask': masks
                }
            )
            
            total_loss = loss_dict['total_loss']
            cls_loss = loss_dict['classification_loss']
            seg_loss = loss_dict['segmentation_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_max_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_seg_loss += seg_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'cls': f"{cls_loss.item():.4f}",
                'seg': f"{seg_loss.item():.4f}"
            })
            
            self.global_step += 1
        
        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_seg_loss = epoch_seg_loss / num_batches
        
        return {
            'train_total_loss': avg_loss,
            'train_cls_loss': avg_cls_loss,
            'train_seg_loss': avg_seg_loss
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_seg_loss = 0.0
        num_batches = len(self.val_loader)
        
        # Progress bar
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move batch to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    predictions=outputs,
                    targets={
                        'label': labels,
                        'mask': masks
                    }
                )
                
                total_loss = loss_dict['total_loss']
                cls_loss = loss_dict['classification_loss']
                seg_loss = loss_dict['segmentation_loss']
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_seg_loss += seg_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'cls': f"{cls_loss.item():.4f}",
                    'seg': f"{seg_loss.item():.4f}"
                })
        
        # Average losses
        avg_loss = epoch_loss / num_batches
        avg_cls_loss = epoch_cls_loss / num_batches
        avg_seg_loss = epoch_seg_loss / num_batches
        
        return {
            'val_total_loss': avg_loss,
            'val_cls_loss': avg_cls_loss,
            'val_seg_loss': avg_seg_loss
        }
    
    def save_checkpoint(
        self,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint to Google Drive.
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'history': self.history,
            'config': self.config.to_dict(),
            'best_metric_value': self.best_metric_value
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint separately
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint['history']
        self.best_metric_value = checkpoint['best_metric_value']
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def fit(self, resume_from: Optional[str] = None):
        """
        Train the model for specified number of epochs.
        
        Args:
            resume_from: Optional path to checkpoint to resume from
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Save config
        config_path = Path(self.config.log_dir) / "training_config.json"
        self.config.save(str(config_path))
        logger.info(f"Saved training config: {config_path}")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            metrics['learning_rate'] = current_lr
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_total_loss'])
            self.history['val_loss'].append(val_metrics['val_total_loss'])
            self.history['train_cls_loss'].append(train_metrics['train_cls_loss'])
            self.history['val_cls_loss'].append(val_metrics['val_cls_loss'])
            self.history['train_seg_loss'].append(train_metrics['train_seg_loss'])
            self.history['val_seg_loss'].append(val_metrics['val_seg_loss'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['train_total_loss']:.4f} "
                       f"(cls: {train_metrics['train_cls_loss']:.4f}, "
                       f"seg: {train_metrics['train_seg_loss']:.4f})")
            logger.info(f"  Val Loss:   {val_metrics['val_total_loss']:.4f} "
                       f"(cls: {val_metrics['val_cls_loss']:.4f}, "
                       f"seg: {val_metrics['val_seg_loss']:.4f})")
            logger.info(f"  LR: {current_lr:.2e}")
            
            # Check if this is the best model
            current_metric_value = metrics[self.config.best_metric]
            is_best = self._is_better(current_metric_value, self.best_metric_value)
            
            if is_best:
                self.best_metric_value = current_metric_value
                self.epochs_without_improvement = 0
                logger.info(f"  ✓ New best {self.config.best_metric}: {current_metric_value:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                if self.config.save_best_only and not is_best:
                    pass  # Skip saving non-best checkpoints
                else:
                    self.save_checkpoint(metrics, is_best=is_best)
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric_value)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if self.config.use_early_stopping:
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                    break
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*70}")
        logger.info(f"Training complete!")
        logger.info(f"Total time: {elapsed_time / 3600:.2f} hours")
        logger.info(f"Best {self.config.best_metric}: {self.best_metric_value:.4f}")
        logger.info(f"{'='*70}\n")
        
        # Save final checkpoint
        final_metrics = {**train_metrics, **val_metrics}
        self.save_checkpoint(final_metrics, is_best=False, filename="final_model.pth")
        
        # Save training history
        history_path = Path(self.config.log_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history: {history_path}")
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.best_metric_mode == 'min':
            return current < best
        else:
            return current > best
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot total loss
            axes[0, 0].plot(self.history['train_loss'], label='Train')
            axes[0, 0].plot(self.history['val_loss'], label='Validation')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot classification loss
            axes[0, 1].plot(self.history['train_cls_loss'], label='Train')
            axes[0, 1].plot(self.history['val_cls_loss'], label='Validation')
            axes[0, 1].set_title('Classification Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot segmentation loss
            axes[1, 0].plot(self.history['train_seg_loss'], label='Train')
            axes[1, 0].plot(self.history['val_seg_loss'], label='Validation')
            axes[1, 0].set_title('Segmentation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot learning rate
            axes[1, 1].plot(self.history['learning_rates'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved training plot: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
