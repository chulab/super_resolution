"""Contains 1D response functions (PSF) to model US devices."""

import numpy as np

from typing import List

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
    center_wavelength: float,
    transducer_bandwidth: float,
    length: int,
    dz: float,
):
  """Simulates a gaussian pulse given a wavelength and bandwidth.

  The center of the pulse is assumed to be at the center index of the output.

  Args:
    center_wavelength: Center wavelength in meters.
    transducer_bandwidth: Bandwidth of transducer used to generate pulse.
    dz: grid size in meters.

  Returns:
    `np.ndarray` of shape `[length]` containing pulse window.
  """
  # Pulse window determines the "pulse" length and shape.
  center_frequency = _SOUND_SPEED_WATER / center_wavelength
  frequency_bandwidth = center_frequency * transducer_bandwidth

  # Convert bandwidth FWHM to standard deviation.
  frequency_sigma = frequency_bandwidth / _FWHM_STANDARD_DEVIATION_RATIO

  # Compute bandwidth limited gaussian pulse.
  pulse_sigma_z = _SOUND_SPEED_WATER / (np.pi * np.sqrt(2) * frequency_sigma)

  # Discretize pulse on grid.
  pulse_sigma_z_grid = pulse_sigma_z / dz

  return utils.discrete_gaussian(length, pulse_sigma_z_grid)



def gaussian_lateral(
    length: int,
    wavelength: float,
    dz: float,
    numerical_aperature: float = .125,
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


def hermite_polynomial(
    value: np.ndarray,
    degree: int
):
    """Computes the physicists' Hermite polynomial of specified degree.

    See `https://en.wikipedia.org/wiki/Hermite_polynomials`

    Args:
      value: `np.ndarray` of any shape.
      degree: `int` denoting degree of Hermite polynomial.

    Returns:
      `np.ndarray` of same shape as `value`.
    """
    coefficients = np.zeros(degree+1)
    coefficients[degree] = 1
    return np.polynomial.hermite.hermval(value, coefficients)


def _rayleigh_length(
    waist_radius,
    wavelength,
):
  return np.pi * waist_radius ** 2 / wavelength


def _complex_beam_parameter(
    z_coordinate,
    rayleigh_length
):
  return z_coordinate + 1j * rayleigh_length


def coordinate_grid(
    lengths: List[float],
    grid_dimensions: List[float],
    center,
):
  """Creates coordinate meshgrids of arbitrary dimension.

  This function returns a list of meshgrids corresponding to an N-D grid where
  N is the length of `lengths` and `grid_dimensions`.

  Args:
    lengths: Lengths of axes in physical units.
    grid_dimensions: Dimensions of grid spacing in same units as `lengths`.
    center: If `True` then the coordinate grid will be centered at the origin.
      If `False` then

  Returns:
    List of coordinate arrays where the `(x_i, ... x_j)`th element of the nth
    array represents the n-coordinate at that element. For more information see
    documentation for `np.meshgrid`.
  """


  if len(lengths) != len(grid_dimensions):
    raise ValueError("`lengths` and `grid_dimensions` must have same number of "
                     "elements")

  coordinates = [np.arange(0, length + step, step) for
                 length, step in zip(lengths, grid_dimensions)]

  if center:
    centers = [length / 2 for length in lengths]
    coordinates = [coor - center for coor, center in zip(coordinates, centers)]

  return np.meshgrid(*coordinates, indexing='ij')


def hermite_gaussian_mode(
    coordinates: np.ndarray,
    wavelength: float,
    L: int,
    M: int = 0,
    numerical_aperature: float = .125,
):
  """Computes complex E(x, y, z) over the volume of a
  `length * dz` x `length * dz` x `length * dz` cube centered at the origin.

  See `https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes`

  Args:
    coordinates: Array of shape `dimensions + [3]` where entries in the last
      contain 3D coordinates `[x, y, z]` at which to evaluate `gaussian`.
    wavelength: Float denoting wavelength in meters.
    dz: float denoting grid size in meters.
    L : int denoting L order of hermite-gaussian.
    M : int denoting M order of hermite-gaussian.
    numerical_aperture: float denoting NA of collecting apparatus.
    amplitude: float denoting amplitude of beam (specifically, E(0, 0, 0)).

  Returns:
    3D `np.ndarray` containing `E(x, y, z)`.
  """
  if coordinates.shape[-1] != 3:
    raise ValueError("Last dimension of `coordinates` must be 3")

  x = coordinates[..., 0]
  y = coordinates[..., 1]
  z = coordinates[..., 2]

  waist_radius = beam_waist_radius(wavelength, numerical_aperature)

  rayleigh_length = _rayleigh_length(waist_radius, wavelength)
  waist_z = waist_radius * np.sqrt(1 + (z / rayleigh_length) ** 2)

  axial_amplitude = waist_radius / waist_z

  hermite_polynomial_L = hermite_polynomial(np.sqrt(2) * x / waist_z, L)
  hermite_polynomial_M = hermite_polynomial(np.sqrt(2) * y / waist_z, M)

  transverse_amplitude = np.exp(-1 * (x ** 2 + y ** 2) / waist_z ** 2)

  gouy_phase = (L + M + 1) * np.arctan(z / rayleigh_length)

  phase_component = np.exp(1j * gouy_phase)

  return (axial_amplitude * hermite_polynomial_L * hermite_polynomial_M *
    transverse_amplitude * phase_component)


def gaussian_impulse_response(
    coordinates,
    frequency,
    mode,
    numerical_aperture,
    bandwidth
):
  """Constructs a PSF of a guassian pulse centered on coordinate grid.

  This function constructs the 2D impulse response of a gaussian pulse where
  `bandwidth` < 1.

  Args:
    coordinates: `np.ndarray` of shape `[X, Y, Z, 3]` where the `[X, Y, Z]`th
      slice contains the `X`, `Y`, and `Z` coordinates.
    frequency: frequency of wave in Hz.
    mode: Gaussian L mode.
    numerical_aperture: NA of US device.
    bandwidth: See `gaussian_pulse`.

  Returns: np.ndarray of shape `coordinates.shape[:-1]` containing
    gaussian pulse sampled at points given by `coordinates`.
  """
  wavelength = defs.wavelength_from_frequency(frequency)
  wavenumber = np.pi * 2 / wavelength

  # We first generate the amplitude of a monochromatic gaussian beam.
  mode_amplitude = hermite_gaussian_mode(
    coordinates=coordinates,
    wavelength=wavelength,
    L=mode,
    M=0,
    numerical_aperature=numerical_aperture
  )

  # Calculate the monochromatic instantaneous phase term.
  spatial_phase = np.exp(-1j * wavenumber * coordinates[..., 2])

  # Compute grid size in z-axis.
  dz = coordinates[0, 0, 1, 2]-coordinates[0, 0, 0, 2]

  # Compute windowing amplitude which is applied to the gaussian beam.
  pulse_window = gaussian_pulse(
    center_wavelength=wavelength,
    transducer_bandwidth=bandwidth,
    length=mode_amplitude.shape[-1],
    dz=dz,
  )

  # Real component of beam is physical impulse response. We slice along the
  # center coodinate in the `Y` direction as we use a 1D array.
  return np.real(mode_amplitude * spatial_phase * pulse_window)



def frequency_bandwidth_with_gaussian_noise(
    length: int,
    dz: float,
    center_frequency: float,
    frequency_bandwidth: float,
    gaussian_sigma: float,
    amplitude: float = 1.,
):
    """Returns psf along propagation axis due to beam obtained from inverse-
    fourier transform of a frequency bandwidth, convolved with a normalized
    gaussian noise, both around center_frequency.

    Args:
        length: int denoting number of pixels.
        dz: float denoting grid size in meters.
        center_frequency: frequency of center of bandwidth in Hz.
        frequency_bandwidth: difference between max and min frequency in Hz.
        gaussian_sigma: standard deviation of gaussian noise.
        amplitude: float denoting amplitude of beam (at z = 0).

    Returns:
    `np.ndarray` of shape `[length]`.

    The inverse fourier transform of f * g is simply the multiplication of the
    inverse fourier transform of f with the inverse fourier transform of g.

    The inverse fourier transform of a rectangular pulse is sinc so the inverse
    transform of the frequency bandwidth can be shown to be
    cos(2pi f_ct) sinc(bt) where f_c is the center_frequency and b is the
    frequency bandwidth.

    The inverse fourier transform of the gaussian noise centered about f_c is
    cos(2pi f_ct) e^{-2pi^2 gaussian_sigma^2 t^2}.
    """

    if length % 2 != 1:
      raise ValueError("`length` must be odd, but got {}".format(length))

    center_z = length // 2 * dz
    z = np.linspace(-center_z, center_z, length)
    t = z / _SOUND_SPEED_WATER

    inverse_freq = np.cos(2 * np.pi * center_frequency * t) * np.sinc(frequency_bandwidth * t)
    inverse_gaussian = np.cos(2 * np.pi * center_frequency * t) * np.exp(-2 * (np.pi * gaussian_sigma * t) ** 2)

    normalized = inverse_freq * inverse_gaussian

    return  amplitude * normalized / normalized[length//2]
