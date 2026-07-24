"""
Checkpoint management for VERITAS training.

Handles saving/loading checkpoints to Google Drive with proper versioning.

Requirements: 13.9, 21.7
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
import json
import shutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints on Google Drive.
    
    Features:
    - Save/load checkpoints
    - Keep best N checkpoints
    - Automatic versioning
    - Checkpoint metadata
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        keep_best_n: int = 3
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (Google Drive)
            keep_last_n: Number of recent checkpoints to keep
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized CheckpointManager at {checkpoint_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'checkpoints': [],
            'best_checkpoints': [],
            'latest_checkpoint': None
        }
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        additional_info: Optional[Dict] = None
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is a best checkpoint
            additional_info: Additional information to save
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / filename
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Update metadata
        checkpoint_info = {
            'filename': filename,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        self.metadata['latest_checkpoint'] = filename
        
        if is_best:
            self.metadata['best_checkpoints'].append(checkpoint_info)
            # Sort best checkpoints by metric
            self.metadata['best_checkpoints'].sort(
                key=lambda x: x['metrics'].get('val_total_loss', float('inf'))
            )
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint dictionary
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        return self.load_checkpoint(str(best_path), model, optimizer, scheduler, device)
    
    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """Load latest checkpoint."""
        if not self.metadata['latest_checkpoint']:
            raise FileNotFoundError("No checkpoints found")
        
        latest_path = self.checkpoint_dir / self.metadata['latest_checkpoint']
        return self.load_checkpoint(str(latest_path), model, optimizer, scheduler, device)
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints."""
        # Keep only last N checkpoints
        if len(self.metadata['checkpoints']) > self.keep_last_n:
            # Get checkpoints to remove (not best ones)
            checkpoints_to_remove = []
            for cp in self.metadata['checkpoints'][:-self.keep_last_n]:
                if not cp['is_best']:
                    checkpoints_to_remove.append(cp)
            
            # Remove files
            for cp in checkpoints_to_remove:
                checkpoint_path = self.checkpoint_dir / cp['filename']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {cp['filename']}")
                
                # Remove from metadata
                self.metadata['checkpoints'].remove(cp)
        
        # Keep only best N best checkpoints
        if len(self.metadata['best_checkpoints']) > self.keep_best_n:
            worst_best = self.metadata['best_checkpoints'][self.keep_best_n:]
            for cp in worst_best:
                # Don't delete file if it's in recent checkpoints
                if cp not in self.metadata['checkpoints'][-self.keep_last_n:]:
                    checkpoint_path = self.checkpoint_dir / cp['filename']
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        logger.debug(f"Removed old best checkpoint: {cp['filename']}")
            
            # Keep only top N
            self.metadata['best_checkpoints'] = self.metadata['best_checkpoints'][:self.keep_best_n]
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        return self.metadata['checkpoints']
    
    def list_best_checkpoints(self) -> List[Dict]:
        """List best checkpoints."""
        return self.metadata['best_checkpoints']
    
    def get_latest_epoch(self) -> int:
        """Get epoch of latest checkpoint."""
        if self.metadata['checkpoints']:
            return self.metadata['checkpoints'][-1]['epoch']
        return 0


# Backward compatibility alias
CheckpointSaver = CheckpointManager
