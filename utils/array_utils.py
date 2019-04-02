"""Array utility functions."""

import numpy as np

from typing import List

import tensorflow as tf

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

def normalize_along_axis(array: np.ndarray, axis: int=-1, order: int=2):
  """Normalizes n-dimensional array along `axis` according to `order`-norm."""

  norms = np.linalg.norm(array, order, axis)

  # set entries to one if norm is zero so vector is still zero after divison
  if norms.ndim == 0:
      # array is vector
      norms = 1 if norms == 0 else norms
  else:
      # non-trivial array
      norms[norms==0] = 1

  return array / np.expand_dims(norms, axis)

def sample_spherical(count: int, ndim: int):
    """Uniformly samples `count` points on ndim-unit sphere."""
    vec = np.random.randn(ndim, count)
    vec /= np.linalg.norm(vec, axis=0)
    return np.swapaxes(vec, 0, 1)
