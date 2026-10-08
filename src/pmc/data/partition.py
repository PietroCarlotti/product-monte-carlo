"""Deterministic partitioning helpers for federated experiments."""

from __future__ import annotations

from typing import Iterable


def contiguous_partitions(length: int, n_partitions: int) -> list[range]:
    """Split ``length`` rows into equal, deterministic contiguous partitions."""
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(n_partitions, bool) or not isinstance(n_partitions, int):
        raise ValueError("n_partitions must be a positive integer")
    if n_partitions <= 0 or length % n_partitions:
        raise ValueError("length must be divisible by n_partitions")
    width = length // n_partitions
    return [range(i * width, (i + 1) * width) for i in range(n_partitions)]
