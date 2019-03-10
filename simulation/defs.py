"""Common defintions."""

from collections import namedtuple
from typing import Tuple

import numpy as np

_SPEED_OF_SOUND_WATER = 1498  # m/s
_SPEED_OF_SOUND_TISSUE = 1540 # m/s

def frequency_from_wavelength(wavelength):
  """Computes `frequency` in Hz given `wavelength` in meters."""
  return _SPEED_OF_SOUND_WATER / wavelength


def wavelength_from_frequency(frequency):
  """Computes `wavelength` in meters given `frequency` in Hz."""
  return _SPEED_OF_SOUND_WATER / frequency


class ObservationSpec(namedtuple(
  'ObservationSpec', ['angles', 'frequencies', 'modes', 'grid_dimension',
                      'transducer_bandwidth', 'numerical_aperture'])):
  """ObservationSpec contains parameters associated with US observation."""

  def __new__(cls, angles, frequencies, modes, grid_dimension,
              transducer_bandwidth, numerical_aperture):
    assert isinstance(angles, list)
    if not all(0. <= angle < np.pi for angle in angles):
      raise ValueError("All angle in `angles` must be scalars between 0 and "
                       "pi. Got {}.".format(angles))

    assert isinstance(frequencies, list)

    assert isinstance(grid_dimension, float)

    assert isinstance(modes, list)
    if not all(0 <= mode and isinstance(mode, int) for mode in modes):
      raise ValueError("Modes must be integers greater than or equal to 0.")

    assert isinstance(transducer_bandwidth, float)

    assert isinstance(numerical_aperture, float)

    return super(ObservationSpec, cls).__new__(
      cls, angles, frequencies, modes, grid_dimension, transducer_bandwidth,
      numerical_aperture)


class PsfDescription(namedtuple('PsfDescription',
             ['frequency', 'mode', 'frequency_sigma', 'numerical_aperture'])):
  """Contains description of PSF."""

  def __new__(cls, frequency: float, mode: int, frequency_sigma: float,
              numerical_aperture: float):
    assert isinstance(frequency, float)
    assert 0 <= frequency
    assert isinstance(mode, int)
    assert 0<=mode
    assert isinstance(frequency_sigma, float)
    assert 0 <= frequency_sigma
    assert isinstance(numerical_aperture, float)
    assert 0 <= numerical_aperture

    return super(PsfDescription, cls).__new__(
      cls, frequency, mode, frequency_sigma, numerical_aperture)


class PSF(namedtuple('PSF', ['psf_description', 'physical_size', 'array'])):
  """Contains PSF and description"""

  def __new__(cls, psf_description:PsfDescription, physical_size: Tuple[float],
              array: np.ndarray):
    assert isinstance(psf_description, PsfDescription)
    assert len(physical_size) == 2
    assert all(s > 0 for s in physical_size)
    assert array.ndim == 2
    return super(PSF, cls).__new__(cls, psf_description, physical_size, array)