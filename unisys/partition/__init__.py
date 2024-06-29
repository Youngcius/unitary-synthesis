"""Circuit partitioning"""

from .sequential import sequential_partition
from .greedy import greedy_partition

__all__ = [
    'sequential_partition',
    'greedy_partition',
]
