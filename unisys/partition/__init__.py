"""Circuit partitioning"""

from .quick import quick_partition
from .greedy import greedy_partition

__all__ = [
    'quick_partition',
    'greedy_partition',
]
