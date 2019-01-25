"""Constructs filters in Fourier domain."""

from typing import List

import numpy as np

from frequency import utils
from utils import array_utils


def gaussian_filter(
    index: np.ndarray,
    frequency: List[float],
    sigma: List[float],
):
  """Convenience function to generate a set of filters for signal extraction.

  Generates a set of gaussian filters defined by center frequencies and sigma
  values. The filters include both positive and negative components.

  Args:
    index: Array of shape `[filter_length]` containing frequency values at which
      to generate the gaussian filters. Typically this will be of the same shape
      as the image with which the filter will be multiplied.
    frequency: List of length `filter_count` containing center frequencies
      for filters.
    sigma: List of length  `filter_count` containing sigma values
      corresponding to entries in `frequency`.

  Returns:
    Array of shape `[filter_length, filter_count]`

  Raises:
    ValueError: If input has incorrect shape.
  """
  if len(frequency) != len(sigma):
    raise ValueError("`Frequency` and `sigma` should have same length.")

  filters = []
  for freq, sig in zip(frequency, sigma):
    filters.append(utils._gaussian(index, freq, sig) +
                   utils._gaussian(index, -freq, sig))
  return np.stack(filters, -1)


def extract_frequency(
    signal,
    filters,
):
  """Extracts frequency band(s) from signal given filters.

  Args:
    signal: Complex `np.ndarray` of shape `batch_dimensions + [filter_length]`
     representing US image.
    filters: Array of shape `batch_dimensions + [filter_length, filter_count]`.

  Returns:
    `tf.Tensor` of shape `batch_shape + [filter_length, filter_count]`.

  Raises:
    ValueError: If input has incompatible shape.
  """
  if signal.shape[-1] != filters.shape[-2]:
    raise ValueError("`signal` and `filters` must have same `filter_length`"
                     "dimension. Got {} and {}.".format(
      signal.shape[-1], filters.shape[-2]))

  if not array_utils.is_broadcast_compatible(
      signal.shape[:-1], filters.shape[:-2]):
    raise ValueError("`signal` and `filters` must have compatible batch "
                     "dimensions.")

  # Now in frequency domain with DC component at center of signal.
  signal_ft = np.fft.fftshift(np.fft.fft(signal, axis=-1))

  # Add axis for filter multiplication.
  signal_ft = signal_ft[..., np.newaxis]

  # Multiply signal by filter to extract different bands.
  signal_ft = signal_ft * filters

  # Move DC component to 0 index and return to time domain.
  return np.fft.ifft(np.fft.fftshift(signal_ft, axes=-2), axis=-2)