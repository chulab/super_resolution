"""Contains 1D response functions (PSF) to model US devices."""

import numpy as np
from simulation import utils

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
  return np.exp( -points ** 2 / (waist_radius ** 2) )


def hermite_polynomial(
    value: np.ndarray,
    degree: int
):
    """Computes the physicists' Hermite polynomial of specified degree.

    See `https://en.wikipedia.org/wiki/Hermite_polynomials`

    Args:
      value: `np.ndarray` of any shape.
      degree: int denoting degree of Hermite polynomial.

    Returns:
      `np.ndarray` of same shape as value.
    """
    coefficients = np.zeros(degree+1)
    coefficients[degree] = 1
    return np.polynomial.hermite.hermval(value, coefficients)


def _u_hermite_gaussian(
    beam: np.ndarray,
    degree: int,
    waist_radius: float,
    wavelength: float
):
    """Computes u_J(x, z) given in
    `https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes`.

    Args:
      beam: 1. Cross-section of beam
            `np.ndarray` of shape `[length, length, 2]` whose (i, j)th element
            stores the (x, z) coordinates.
            2. Full beam
            `np.ndarray` of shape `[length, length, length, 2]` whose
            (i, j, k)th element stores the (x, z) coordinates.
      degree: int denoting degree of hermite-gaussian.
      waist_radius: float denoting w(0) in meters.
      wavelength: float denoting wavelength in meters.

    Returns:
      According to options for beam above,
      1. Cross-section: `np.ndarray` of shape `[length, length]` for u_J(x, z).
      2. Full: `np.ndarray` of shape `[length, length, length]` for u_J(x, z).
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
    hermite = hermite_polynomial(np.sqrt(2) * x / waist_z, degree)
    amplitude_decay = np.exp(-1j * np.pi / wavelength * (x**2) / complex_beam)

    return norm_constant * norm_z * phase_shift * hermite * amplitude_decay


def coordinate_grid_3d(
    length: int,
    dz: float,
):
    """ Creates an evenly spaced [length, length, length] 3-d grid with each
    element as its corresponding [x, y, z] coordinates.

    See `https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy`

    Args:
      length: int denoting number of pixels.
      dz: float denoting grid size in meters.

    Returns:
      `np.ndarray` of shape `[length, length, length, 3]` where last entry is
      [x, y, z] coordinate corresponding to point [length, length, length] in
      the physical grid.
    """

    center = length // 2 * dz

    #If center = 1 and center = 3,
    #X = [[[-1, -1, -1] x 3] [[0, 0, 0] x 3] [[1, 1, 1] x 3]]
    #Y = [[[-1, -1, -1] [0, 0, 0] [1, 1, 1]] x 3]
    #Z = [[[-1, 0, 1] x3] x3 ]
    #X.flatten() = [ -1 x 9, 0 x 9, 1 x 9]
    #Y.flatten() = [ -1 -1 -1 0 0 0 1 1 1 x 3]
    #Z.flatten() = [-1 0 1 x 9]
    #XYZ = combine above as columns and then transpose
    #which is equivalent to combining columns of above.
    #e.g. [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1] ...  ]
    X, Y, Z = np.mgrid[-center:center:length*1j, -center:center:length*1j, -center:center:length*1j];
    XYZ = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    #reshape XYZ which is 1-d to 3-d.
    coordinate_grid = XYZ.reshape(length, length, length, -1)

    return coordinate_grid


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
    """Computes complex E(x, y, z) over the volume of a
    `length * dz` x `length * dz` x `length * dz` cube centered at the origin.

    See `https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes`

    Args:
      length: int denoting number of pixels.
      wavelength: float denoting wavelength in meters.
      dz: float denoting grid size in meters.
      L : int denoting L order of hermite-gaussian.
      M : int denoting M order of hermite-gaussian.
      numerical_aperture: float denoting NA of collecting apparatus.
      amplitude: float denoting amplitude of beam (specifically, E(0, 0, 0)).

    Returns:
      `np.ndarray` of shape `[length, length, length]` which stores E(x, y, z).
    """

    if length % 2 != 1:
      raise ValueError("`length` must be odd, but got {}".format(length))

    beam = coordinate_grid_3d(length ,dz)
    waist_radius = beam_waist_radius(wavelength, numerical_aperature)

    #extract x, z coordinates from beam.
    u_xz = _u_hermite_gaussian(beam[:, :, :, [0, 2]], L, waist_radius, wavelength);

    #extract y, z coordinates from beam.
    u_yz = _u_hermite_gaussian(beam[:, :, :, [1, 2]], M, waist_radius, wavelength);
    normalized = u_xz * u_yz #physically normalized beam which may be useful

    #scale normalized beam such that E(0, 0, 0) is amplitude.
    return amplitude * normalized / normalized[length//2, length//2, length//2]

def gaussian_axial_mode(
    length: int,
    wavelength: float,
    dz: float,
    degree: int = 0,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
    """Returns psf for gaussian beam along propagation axis."""

    return (hermite_gaussian_mode(length, wavelength, dz, degree, 0, numerical_aperature, amplitude)[length//2, length//2, :])

def gaussian_lateral_mode(
    length: int,
    wavelength: float,
    dz: float,
    degree: int = 0,
    numerical_aperature: float = .125,
    amplitude: float = 1.,
):
    """Returns 1D lateral psf with discrete gaussian profile."""

    return (hermite_gaussian_mode(length, wavelength, dz, degree, 0, numerical_aperature, amplitude)[:, length//2, length//2])

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
