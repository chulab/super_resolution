"""Contains 1D response functions (PSF) to model US devices."""

import tensorflow as tf
import numpy as np
from simulation import utils
from simulation import defs

_FWHM_STANDARD_DEVIATION_RATIO = 2.355
_SOUND_SPEED_WATER = defs._SPEED_OF_SOUND_WATER  # In meters.
_FWHM_1_OVER_E_SQUARED = 1.699

def gaussian_axial(
    length,
    wavelength,
    dz,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
  """Returns psf for gaussian beam along propagation axis.

  For further information see https://en.m.wikipedia.org/wiki/Gaussian_beam.

  Equations from Svelto, Orazio (2010). Principles of Lasers.

  Args:
    length: Length in pixels.
    center_wavelength: Center wavelength in meters.
    dz: grid size in meters.
    numerical_aperture: NA of collecting apparatus.
    amplitude: Amplitude of beam.

  Returns:
    `np.ndarray` of shape `[length]`.

  Raises:
    ValueError: If `length` is not odd.
  """
  if length % 2 != 1:
    raise ValueError("`length` must be odd, but got {}".format(length))

  center_z = length // 2 * dz
  z = np.linspace(-center_z, center_z, length)

  waist_radius = beam_waist_radius(wavelength, numerical_aperature)
  raleigh_length = np.pi * waist_radius ** 2 / wavelength
  waist_z = waist_radius * np.sqrt(1 + (z / raleigh_length) ** 2)
  axial_amplitude =  waist_radius / waist_z

  # Guoy phase.
  gouy_phase = np.arctan(z / raleigh_length)

  wave_number = 2 * np.pi / wavelength
  phase_component = np.exp(-1j * (wave_number * z - gouy_phase))

  return amplitude * axial_amplitude * phase_component


def beam_waist_radius(
    wavelength: float,
    numerical_aperature: float,
):
  """Calculates beam waist based on Abbe limit.

  See `https://en.wikipedia.org/wiki/Diffraction-limited_system# \
   The_Abbe_diffraction_limit_for_a_microscope`
   and `https://en.wikipedia.org/wiki/Beam_diameter#1/e2_width`

  Args:
    wavelength: Wavelength of excitation.
    numerical_aperature:

  Returns:
      Float describing standard deviation of beam at waist (\w_0).
  """
  # Compute FWHM of field. The factor of `\sqrt(2)` accounts for the fact that
  # The Abbe FWHM accounts for the intensity not the field amplitude.
  full_width_half_max_field = (wavelength / (2 * numerical_aperature)
                               * np.sqrt(2))

  # Convert FWHM to 1 / e ^ 2.
  return _FWHM_1_OVER_E_SQUARED * full_width_half_max_field / 2


def gaussian_pulse(
    beam: np.ndarray,
    center_wavelength: float,
    transducer_bandwidth: float,
    dz: float,
):
  """Simulates a gaussian pulse by applying a windowing function to a beam.

  The center of the pulse is assumed to be at the center index of `beam`.

  Args:
    beam: `np.ndarray` of shape `[length]` which models the beam.
    center_wavelength: Center wavelength in meters.
    transducer_bandwidth: Bandwidth of transducer used to generate pulse.
    dz: grid size in meters.

  Returns:
    `np.ndarray` of shape `[length]`.
  """
  # Pulse window determines the "pulse" length and shape.
  center_frequency = _SOUND_SPEED_WATER / center_wavelength
  frequency_bandwidth = center_frequency * transducer_bandwidth

  # convert badwidth FWHM to standard deviation.
  frequency_sigma = frequency_bandwidth / _FWHM_STANDARD_DEVIATION_RATIO

  # Compute bandwidth limited gaussian pulse.
  pulse_sigma_z = _SOUND_SPEED_WATER / (np.pi * np.sqrt(2) * frequency_sigma)

  # Discretize pulse on grid.
  pulse_sigma_z_grid = pulse_sigma_z / dz
  pulse_window = utils.discrete_gaussian(beam.shape[0], pulse_sigma_z_grid)

  # Multiply beam with pulse window.
  return beam * pulse_window


def gaussian_lateral(
    length: int,
    wavelength: float,
    numerical_aperature: float,
    dz: float,
):
  """Returns 1D lateral psf with discrete gaussian profile.

  Computes the lateral field profile of a gaussian beam. Equation comes
  from setting `z =0` in `https://en.m.wikipedia.org/wiki/Gaussian_beam`.

  """
  if length % 2 != 1:
    raise ValueError("`length` must be odd, but got {}".format(length))
  waist_radius = beam_waist_radius(wavelength, numerical_aperature)

  # Convert `waist_radius` to grid units.
  waist_radius = waist_radius / dz

  half_length = length // 2
  points = np.arange(start=-half_length, stop=half_length + 1)
  return np.exp(-points ** 2 / waist_radius ** 2)