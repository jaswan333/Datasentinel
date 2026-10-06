"""
DataSentinel - Poisoning Simulator (Backward Compatibility Shim)
Delegates to detectors.poisoning suite.
"""

from .poisoning.label_flip import flip_labels

__all__ = ["flip_labels"]
