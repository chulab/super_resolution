"""Array utility functions."""

import numpy as np

from typing import List


def is_broadcast_compatible(
    shape_a: List[int],
    shape_b: List[int],
):
  if len(shape_a) != len(shape_b):
    return False
  return all(dim_a == dim_b or dim_a == 1 or dim_b == 1 for dim_a, dim_b in
             zip(shape_a, shape_b))


def reduce_split(array: np.ndarray, axis: int):
  """Splits n-dimensional array along `axis` into `n-1` dimensional chunks."""
  return [np.squeeze(a) for a in np.split(array, array.shape[axis], axis)]
