"""Test for `utils.py`"""

import numpy as np
from simulation import response_functions
from simulation import utils
import unittest


class utilsTest(unittest.TestCase):

  #TODO(noahyt): Write unit test for `gaussian_axial`.

  def testBeamWaist(self):
    wavelength = 6
    numerical_aperature = 1.
    self.assertAlmostEqual(response_functions.beam_waist_radius(
        wavelength, numerical_aperature), 3.60, 2)

  def testGaussianPulse(self):
    beam = np.ones(11)
    center_wavelength = 5e-6
    bandwidth = .8
    dz = center_wavelength / 4
    pulse = response_functions.gaussian_pulse(
      beam, center_wavelength, bandwidth, dz)
    true_pulse = utils.discrete_gaussian(11, 2.6503)
    np.testing.assert_allclose(pulse, true_pulse, 3)


  def testGaussianMode(self):
    """Gaussian with L=M=0 identical to `gaussian_axial` and `gaussian_lateral`."""
    length = 101
    wavelength = 6e-6
    dz = wavelength / 4
    numerical_aperature= .125
    degree=0

    physical_length = length * dz - dz
    coordinates = np.stack(
     response_functions.coordinate_grid(
       [physical_length] * 3, [dz] * 3, center=True), -1)

    axial_pulse = response_functions.gaussian_axial(length, wavelength, dz)
    axial_pulse_mode = response_functions.hermite_gaussian_mode(
      coordinates, wavelength, degree, 0, numerical_aperature
    )[length//2, length//2, :]

    # Check absolute value as quantities are identical to a phase of kpi/4
    np.testing.assert_allclose(
      np.absolute(axial_pulse), np.absolute(axial_pulse_mode), 1e-10)

    lateral_pulse = response_functions.gaussian_lateral(length, wavelength, dz)
    lateral_pulse_mode = response_functions.hermite_gaussian_mode(
      coordinates, wavelength, degree, 0, numerical_aperature
    )[:, length//2, length//2]

    # Check absolute value as quantities are identical to a phase of kpi/4
    np.testing.assert_allclose(
      np.absolute(lateral_pulse), np.absolute(lateral_pulse_mode), 1e-10)

  def testCoordinateGridTest(self):
    lengths = [1, 2, 3]
    grid_dimensions = [.1, .5, 1.]
    xx, yy, zz = response_functions.coordinate_grid(
      lengths, grid_dimensions, center=False)

    np.testing.assert_allclose(
      xx,
      np.tile(np.arange(0, 1.1, .1)[:, np.newaxis, np.newaxis], [1, 5, 4])
    )

    np.testing.assert_allclose(
      yy,
      np.tile(np.arange(0, 2.5, .5)[np.newaxis, :, np.newaxis], [11, 1, 4])
    )

    np.testing.assert_allclose(
      zz,
      np.tile(np.arange(0, 4., 1.)[np.newaxis, np.newaxis, :], [11, 5, 1])
    )


  def coordinateGridTestCenter(self):
    lengths = [1, 2, 3]



if __name__ == "__main__":
  unittest.main()
