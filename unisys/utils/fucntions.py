"""
Other Utils functions
"""

from math import pi


def limit_angle(a: float) -> float:
    """Limit equivalent rotation angle into (-pi, pi]."""
    if a >= 0:
        r = a % (2 * pi)
        if r >= 0 and r <= pi:
            return r
        else:
            return r - 2 * pi
    else:
        r = (-a) % (2 * pi)
        if r >= 0 and r <= pi:
            return -r
        else:
            return 2 * pi - r


def is_power_of_two(num):
    """Check whether a number is power of 2 or not."""
    return (num & (num - 1) == 0) and num != 0
