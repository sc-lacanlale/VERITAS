"""
Data processing and annotation parsing for VERITAS.
"""

from .annotation_parser import (
    AnnotationParser,
    ImageAnnotation,
    parse_openforensics_json
)
from .mask_converter import (
    polygon_to_mask,
    create_mask_from_annotation,
    batch_create_masks
)
from .validator import (
    DataValidator,
    validate_dataset
)
from .splitter import (
    DatasetSplitter,
    create_splits
)
from .dataset import (
    VERITASDataset,
    create_dataloader
)

__all__ = [
    'AnnotationParser',
    'ImageAnnotation',
    'parse_openforensics_json',
    'polygon_to_mask',
    'create_mask_from_annotation',
    'batch_create_masks',
    'DataValidator',
    'validate_dataset',
    'DatasetSplitter',
    'create_splits',
    'VERITASDataset',
    'create_dataloader'
]
