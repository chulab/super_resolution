"""Contains 1D response functions (PSF) to model US devices."""

import numpy as np
import utils

_FWHM_STANDARD_DEVIATION_RATIO = 2.355
_SOUND_SPEED_WATER = 1498  # In meters.
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
  return np.exp(-points ** 2 / (waist_radius ** 2))


def hermite_val(
    value: np.ndarray,
    degree: int
):
    """Computes the physicists' Hermite polynomial of specified degree at values specified

    Reference: https://en.wikipedia.org/wiki/Hermite_polynomials

    Args:
      value: `np.ndarray` of any shape
      degree: int denoting degree of Hermite polynomial

    Returns:
      `np.ndarray` of same shape as value
    """
    coefficients = np.zeros(degree+1)
    coefficients[degree] = 1
    return np.polynomial.hermite.hermval(value, coefficients)


def u_hermite_gaussian(
    beam: np.ndarray,
    degree: int,
    waist_radius: float,
    wavelength: float
):
    """Computes u_J(x, z) given in https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes

    Args:
      beam: Either 1. Cross-section of beam
                      `np.ndarray` of shape `[length]` x `[length]` x 2 whose (i, j)th element stores the (x, z) coordinates
            Or     2. Full beam
                      `np.ndarray` of shape `[length]` x `[length]` x `[length]`x 2 whose (i, j, k)th element stores the (x, z) coordinates
      degree: int denoting degree of hermite-gaussian
      waist_radius: float denoting w(0) in meters, defined in https://en.wikipedia.org/wiki/Gaussian_beam#Beam_waist
      wavelength: float denoting wavelength in meters

    Returns:
      1. Cross-section of beam: `np.ndarray` of shape `[length]` x `[length]` which stores u_J(x, z)
      2. Full beam: `np.ndarray` of shape `[length]` x `[length]` x `[length]` which stores u_J(x, z)
    """

    if (beam.ndim != 3) and (beam.ndim != 4):
        raise ValueError("dimension of `beam` must be 3 or 4, but got dimension {}".format(beam.ndim))

    if (beam.ndim == 3):
        x = beam[:, :, 0];
        z = beam[:, :, 1];
    else:
        x = beam[:, :, :, 0];
        z = beam[:, :, :, 1];

    raleigh_length = np.pi * waist_radius ** 2 / wavelength
    waist_z = waist_radius * np.sqrt(1 + (z / raleigh_length) ** 2)
    complex_beam = z + 1j * raleigh_length
    complex_beam_conjugate = z - 1j * raleigh_length

    norm_constant = np.sqrt(np.sqrt(2/np.pi)/ np.exp2(degree) / np.math.factorial(degree) / waist_radius)
    norm_z = np.sqrt(1j * waist_radius / complex_beam)
    phase_shift = np.power(-complex_beam_conjugate / complex_beam, degree/2)
    hermite = hermite_val(np.sqrt(2) * x / waist_z, degree)
    amplitude_decay = np.exp(-1j * np.pi / wavelength * (x**2) / complex_beam)

    return norm_constant * norm_z * phase_shift * hermite * amplitude_decay


#TODO include different lengths and grid sizes in all three directions
def hermite_gaussian_mode(
    length: int,
    wavelength: float,
    dz: float,
    L: int,
    M: int = 0,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
    """Computes complex E(x, y, z) over the volume of a `length * dz` x `length * dz` x `length * dz` cube
    centered at the origin (measured in meters)

    Reference: https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes

    Args:
      length: int denoting number of pixels
      wavelength: float denoting wavelength in meters.
      dz: float denoting grid size in meters.
      L, M: ints denoting (L, M) order of hermite-gaussian
      numerical_aperture: float denoting NA of collecting apparatus.
      amplitude: float denoting amplitude of beam (more specifically, E(0, 0, 0))

    Returns:
      `np.ndarray` of shape `[length]` x `[length]` x `[length]` which stores E(x, y, z)
    """

    if length % 2 != 1:
      raise ValueError("`length` must be odd, but got {}".format(length))

    center = length // 2 * dz
    """ Creates an evenly spaced `[length]` x `[length]` x `[length]` 3-d grid with each element as [x, y, z] coordinates

    Note: beam is of shape `[length]` x `[length]` x `[length]` x 3 as it is a 3-d grid which stores another 1-d grid
    Reference: https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy

    """
    beam = np.mgrid[-center:center:length*1j, -center:center:length*1j, -center:center:length*1j].reshape(3,-1).T.reshape(length, length, length, -1)
    waist_radius = beam_waist_radius(wavelength, numerical_aperature)
    u_xz = u_hermite_gaussian(beam[:, :, :, [0, 2]], L, waist_radius, wavelength);  #extract x, z coordinates from beam
    u_yz = u_hermite_gaussian(beam[:, :, :, [1, 2]], M, waist_radius, wavelength);  #extract y, z coordinates from beam
    normalized = u_xz * u_yz    #normalized beam which may be useful in the future

    #scale normalized beam such that E(0, 0, 0) is amplitude
    return amplitude * normalized / normalized[length//2, length//2, length//2]

def gaussian_axial_mode(
    length: int,
    wavelength: float,
    dz: float,
    degree: int = 0,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
    return (hermite_gaussian_mode(length, wavelength, dz, degree, 0, numerical_aperature, amplitude)[length//2, length//2, :])

def gaussian_lateral_mode(
    length: int,
    wavelength: float,
    dz: float,
    degree: int = 0,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
    return (hermite_gaussian_mode(length, wavelength, dz, degree, 0, numerical_aperature, amplitude)[:, length//2, length//2])
