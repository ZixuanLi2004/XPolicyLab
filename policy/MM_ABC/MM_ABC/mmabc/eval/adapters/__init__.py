"""Benchmark adapters. Importing the package registers the built-in ones."""

from mmabc.eval.adapters.base import BenchmarkAdapter, available, build_adapter, register
from mmabc.eval.adapters.mobile_arena import MobileArenaAdapter

__all__ = [
    "BenchmarkAdapter",
    "MobileArenaAdapter",
    "available",
    "build_adapter",
    "register",
]
