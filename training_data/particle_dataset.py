"""Functions to make particle dataset for training superresolution models."""

import numpy as np

from typing import Tuple


def particle_distribution(
    size: Tuple[int, int, int],
    distribution_fn: callable = None,
    **kwargs,
):
  """Simulates normalized scatterer distribution.

  This function populates a grid with random values which represent the
  normalized fraction of scatters in that voxel.

  `distribution` sets the function used to generate the random values in each
  voxel.

  Values are clipped outside the range [0, 1).

  Args:
    dimensions: Number of pixels.
    distribution_fn: Function to generate random values. Defaults to
    `numpy.random.rand`.
    kwargs: Arguments passed to `distribution_fn`.

  Returns:
    np.ndarray of shape `size`.

  Raises:
    ValueError: If arguments are invalid.
  """
  if len(size) != 3:
    raise ValueError("`size` must be a tuple of 3 `int`."
                     "Got {}".format(size))
  if distribution_fn is None:
    distribution_fn = np.random.random

  distribution = distribution_fn(size=size, **kwargs)

  return np.clip(distribution, 0, 1.)


def poisson_noise(
    array: np.ndarray,
    lambda_multiplier: int = 500,
    normalize_output: bool = True,
):
  """Adds poissonian noise to a distribution and optionally normalizes output.

  This function is used to add noise to scatterer distributions. It functions by
  first multiplying the distribution (which is assumed to take values between
  0. and 1.) by the `lambda_multiplier` to generate the true expected number of
  scatterers in a region. Then, for each pixel, a value is drawn from a poisson
  parametrized by the given lambda. This generates an array containing the
  integer number of scatterers present in each pixel. The output may be
  normalized again before returning.

  Args:
    array: `np.ndarray` of shape `[batch_size] + physical_dimensions` containing
      distribution of scatterers (normalized between 0. and 1.).
    lambda_multiplier: Multiplier used with `array` to determine expected
      value (number) of scatterers at each location.  Larger values generate
      more smooth distributions.
    normalize_output: bool. If true then output is normalized between 0 and 1.
      Defaults to `True`. This normalization is applied accross all dimensions
      except the batch axis.

  Returns:
    `np.ndarray` of same size as `array`.
  """
  array = np.random.poisson(array * lambda_multiplier)
  if normalize_output and np.sum(array) != 0:
    return array / np.amax(
      array, axis=tuple(range(array.ndim)[1:]), keepdims=True)
  else:
    return array


def poisson_generator(
    array_generator,
    lambda_multiplier,
    normalize_output,
):
  """Convenience function to apply poisson noise to generator."""
  while True:
    yield(poisson_noise(
      next(array_generator)[np.newaxis], lambda_multiplier, normalize_output)
    )[0]