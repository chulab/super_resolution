"""Utility functions for frequency space of images."""

import numpy as np


def _pad(
    signal: np.ndarray,
    pad_length: int,
):
  """Adds zero-padding along last axis of `signal`."""
  assert isinstance(pad_length, int)

  pad = [[0, 0]] * signal.ndim
  pad[-1] = [0, pad_length]
  return np.pad(signal, pad, mode='constant')


def _fft(
    samples: np.ndarray,
    sample_frequency: float,
    fftshift: bool=True,
):
  """Computes FFT of time signal and frequency axis for plotting results.

  Args:
    samples: Array of shape `[..., sample_count]` of samples, where the indices
      in the last axis correspond to sampling of some function at discrete time:
      `(f(t_0), f(t_0 + 1 / sample_frequency), ...)`.
    sample_frequency: Frequency of measurements used to compute `samples`.
      Measuered in Hz.

  Returns:
    fft: Array of same shape as `samples` containing FFT of last axis of
      `samples`. Note that the fft is complex-valued.
    frequency_index: Array of shape `[sample_count]` containing the frequency at
      indices corresponding to `fft` output.

  Raises:
    ValueError: If input has incorrect type.
  """
  if samples.ndim < 1:
    raise ValueError("`samples` must be at least 1D.")
  if not isinstance(sample_frequency, float):
    raise ValueError("`sample_frequency` must be float.")

  fft = np.fft.fft(samples, axis=-1)
  frequency_index = np.fft.fftfreq(samples.shape[-1], sample_frequency)
  if fftshift:
    frequency_index = np.fft.fftshift(frequency_index)
    fft = np.fft.fftshift(fft)
  return fft, frequency_index


def _gaussian(
    coordinates: np.ndarray,
    center: float,
    sigma: float,
):
  """Evaluates gaussian parametrized by `center`, `sigma` at `coordinates`.

  Typically `coordinates` will be a 1D array of values corresponding to the
  domain of interest, for instance an evenly spaced list of frequency values.
  This function can be used to generate a gaussian curve. Explicitly:

    # x_axis_coordinates = np.linspace(0, 10, 100)
    # y_axis_values = _gaussian(x_axis, 1, .5)
    # plt.plot(x_axis, y_axis)

  Note that this returns the values sampled from a continuous gaussian
  rather than using a discrete gaussian.

  Args:
    coordinates: Array containing scalar values.
    center: Float describing center of gaussian. Must have same units as
      `coordinates.`
    sigma: Float describing width of gaussian. Must have same units as
      `coordinates.`

  Returns:
    `np.ndarray` of same shape as `coordinates` containing gaussian evaluated
    at the values provided in `coordinates`.
  """
  return np.exp(-(coordinates - center) ** 2. / (2 * sigma ** 2.))

