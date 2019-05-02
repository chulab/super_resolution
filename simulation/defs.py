"""Common defintions."""

from collections import namedtuple
from typing import Union

import tensorflow as tf

import numpy as np


_SPEED_OF_SOUND_WATER = 1498  # m/s
_SPEED_OF_SOUND_TISSUE = 1540  # m/s


def frequency_from_wavelength(wavelength):
  """Computes `frequency` in Hz given `wavelength` in meters."""
  return _SPEED_OF_SOUND_WATER / wavelength


def wavelength_from_frequency(frequency):
  """Computes `wavelength` in meters given `frequency` in Hz."""
  return _SPEED_OF_SOUND_WATER / frequency


class PsfDescription(
  namedtuple('PsfDescription',
             ['frequency', 'mode', 'frequency_sigma', 'numerical_aperture'])):
  """Contains description of PSF."""

  def __new__(cls, frequency: float, mode: int, frequency_sigma: float,
              numerical_aperture: float):
    assert isinstance(frequency, float)
    assert 0 <= frequency
    assert isinstance(mode, int)
    assert 0 <= mode
    assert isinstance(frequency_sigma, float)
    assert 0 <= frequency_sigma
    assert isinstance(numerical_aperture, float)
    assert 0 <= numerical_aperture

    return super(PsfDescription, cls).__new__(
      cls, frequency, mode, frequency_sigma, numerical_aperture)


class PSF(namedtuple('PSF', ['psf_description', 'angle', 'array'])):
  """Contains PSF and description"""

  def __new__(
    cls,
    psf_description: defs.PsfDescription,
    angle: float,
    array: Union[np.ndarray, tf.Tensor]
  ):
    assert isinstance(psf_description, defs.PsfDescription)
    if isinstance(array, np.ndarray):
      assert array.ndim == 2
    else:  # Is `Tensor`.
      assert len(array.shape.as_list()) == 2
    return super(PSF, cls).__new__(cls, psf_description, angle, array)


class ObservationSpec(namedtuple(
  'ObservationSpec', ['grid_dimension', 'angles', 'psf_descriptions'])):
  """ObservationSpec contains parameters associated with US observation."""

  def __new__(cls, grid_dimension, angles, psf_descriptions):
    assert isinstance(grid_dimension, float)
    if grid_dimension <= 0:
      raise ValueError("`grid_dimension` must be greater than 0.")

    assert isinstance(angles, list)
    if not all(0. <= angle < np.pi for angle in angles):
      raise ValueError("All angle in `angles` must be scalars between 0 and "
                       "pi. Got {}.".format(angles))

    assert isinstance(psf_descriptions, list)
    if not all(isinstance(description, PsfDescription) for description in
               psf_descriptions):
      raise ValueError("All elements in `psf_descriptions` must be "
                       "`PsfDescription`.")

    return super(ObservationSpec, cls).__new__(
      cls, grid_dimension, angles, psf_descriptions)


class USImage(
  namedtuple('USImage', ['image', 'angle', 'psf_description'])):

  def __new(cls, image: np.ndarray, angle: float,
            psf_description: PsfDescription):
    assert image.ndim == 2
    assert 0. <= angle < np.pi
    assert isinstance(psf_description, PsfDescription)

    return super(USImage, cls).__new__(cls, image, angle, psf_description)
