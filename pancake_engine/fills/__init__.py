"""Fill-model registry (Wave 2, 0.9.0).

Public surface: ``FillModel`` protocol, ``EntryFill`` result, ``resolve()`` lookup.
"""

from .registry import EntryFill, FillModel, resolve

__all__ = ["EntryFill", "FillModel", "resolve"]
