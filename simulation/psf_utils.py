"""Utilities for constructing psf filters."""

import numpy as np

from typing import List

from simulation import defs
from simulation import response_functions
from simulation import estimator

_FROM_SAME = "FROM_SAME"
_FROM_SINGLE = "FROM_SINGLE"
_LATERAL = "LATERAL"
_AXIAL = "AXIAL"


def to_filter(
    psfs: List[np.ndarray],
    mode: str,
) -> np.ndarray:
  """Constructs a single filter from a list of psfs.
  
  This function builds a filter of shape `[length, channels_in, channels_out]`.

  In mode `FROM_SINGLE` this constructs a filter with one input channel and
  `len(psfs)` output channels.

  In mode `FROM_SAME`, the number of input and output channels are the same and
  each psf represents a map from one channel to the same channel in the output.
   In other words,
    F_{l, i, j} = {
        if i == j: psf[l]
        if i != j: [0]
      }

  This is useful when applying multiple filters (for example, when trying
  to simulate multiple observation frequencies) in an observation.

  Args:
    psfs: List of 1D arrays representing the PSF of an imaging device.
    mode: String representing type of filter. Must be one of `FROM_SINGLE`,
    `FROM_SAME`.

  Returns:
    Array of shape `[length, channels_in, channels_out]`.

  Raises:
    ValueError: If all PSF are not the same length. Or mode arg is invalid.
    """
  if any(len(psf.shape) > 1 for psf in psfs):
    raise ValueError("All PSF's must be 1D, got {}.".format(
      [psf.shape for psf in psfs]))
  if any(psf.shape != psfs[0].shape for psf in psfs):
    raise ValueError("All PSF's must have same shape, got {}.".format(
      [psf.shape for psf in psfs]
    ))

  # Stack filters and add `in_channel` dimension.
  filter = np.stack(psfs, -1)[..., np.newaxis, :]

  if mode == _FROM_SINGLE:
    return filter
  if mode == _FROM_SAME:
    # Tile `filter` so `in_channels` has same dimension as `out_channels`.
    filter = np.tile(filter, [1, len(psfs), 1])
    return filter * np.eye(len(psfs))[np.newaxis, ...]
  else:
    raise ValueError("`mode` must be one of {} but got `{}`".format(
      [_FROM_SAME, _FROM_SINGLE], mode))


def lateral_psf_filters(
    psf_length: int,
    wavelengths: List[float],
    numerical_aperture: float,
    grid_dimension: float,
):
  """Builds lateral psf filter(s)."""
  if psf_length % 2 == 0:
    raise ValueError("`psf_length` must be odd.")
  psf_lateral = [response_functions.gaussian_lateral(
    psf_length, wavelength, numerical_aperture, grid_dimension)
    for wavelength in wavelengths]
  return to_filter(psf_lateral, mode="FROM_SINGLE")


def axial_psf_filters(
    psf_length,
    wavelengths,
    numerical_aperture,
    bandwidth,
    grid_dimension,
):
  """Constructs axial psf filter(s)."""
  if psf_length % 2 == 0:
    raise ValueError("`psf_length` must be odd.")
  psf_axial = [
    response_functions.gaussian_pulse(wavelength, bandwidth, psf_length,
                                      grid_dimension) *
    response_functions.gaussian_axial(
      psf_length, wavelength, grid_dimension, numerical_aperture
      )
    for wavelength in wavelengths]
  return to_filter(psf_axial, mode="FROM_SAME")

def psf_filter(
    type: str,
    length: int,
    observation_spec: defs.ObservationSpec,
):
  """Convenience function to construct psf filter from `ObservationSpec`."""
  wavelengths = [defs._SPEED_OF_SOUND_WATER / freq for freq in
                 observation_spec.frequencies]
  if type == _LATERAL:
    return lateral_psf_filters(length, wavelengths, observation_spec.numerical_aperture, observation_spec.grid_dimension,)
  if type == _AXIAL:
    return axial_psf_filters(length, wavelengths, observation_spec.numerical_aperture, observation_spec.transducer_bandwidth, observation_spec.grid_dimension)
  else:
    raise ValueError("`type` must be one of {}, got {}".format([_LATERAL, _AXIAL], type))