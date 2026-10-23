"""Benchmark adapters. Importing the package registers the built-in ones."""

from mmabc.eval.adapters.base import BenchmarkAdapter, available, build_adapter, register
from mmabc.eval.adapters.miverse import MiVerseAdapter

__all__ = [
    "BenchmarkAdapter",
    "MiVerseAdapter",
    "available",
    "build_adapter",
    "register",
]
