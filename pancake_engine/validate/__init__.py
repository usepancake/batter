"""Validation layer for Pancake Engine 0.3.

Errors block the run. Warnings surface but never block. See architecture
§Validation layer for the full check matrix.
"""

from .dataset import validate_dataset
from .macro import validate_reference_dataset
from .spec import validate_spec
from .verdict import ValidationError, ValidationVerdict, ValidationWarning

__all__ = [
    "ValidationError",
    "ValidationVerdict",
    "ValidationWarning",
    "validate_dataset",
    "validate_reference_dataset",
    "validate_spec",
]
