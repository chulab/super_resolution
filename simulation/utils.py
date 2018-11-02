"""Utilities for simulation."""

import math
import numpy as np
from scipy import special

def discrete_gaussian(
    size: int,
    t: float,
):
  """Implements discrete gaussian kernal.

  For further information see:
    https://en.wikipedia.org/wiki/Scale_space_implementation \
    #The_discrete_Gaussian_kernel

  Args:
    size: Length of desired output.
    t: . Equivalent to \sigma in the continuous gaussian distribution.

  Returns:
    `np.ndarray` of shape `[size + 1]`.

  Raises:
    ValueError: If `size` is not odd.
  """
  if size % 2 !=1:
    raise ValueError("`size` must be odd. Got {}".format(size))
  half_size = (size - 1) // 2
  n = np.arange(-half_size, half_size + 1)
  print(special.iv(n, t))
  return math.exp(-t) * special.iv(n, t)