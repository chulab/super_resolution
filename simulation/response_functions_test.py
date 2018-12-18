"""Test for `utils.py`"""

import numpy as np
from scipy import signal
from simulation import response_functions
from simulation import utils
import unittest


class utilsTest(unittest.TestCase):

  #TODO(noahyt): Write unit test for `gaussian_axial`.

  def testBeamWaist(self):
    wavelength = 6e-6
    numerical_aperature = 1.
    self.assertAlmostEqual(response_functions.beam_waist_radius(
        wavelength, numerical_aperature), 1.80e-6, 3)

  def testGaussianPulse(self):
    beam = np.ones(11)
    center_wavelength = 5e-6
    bandwidth = .8
    dz = center_wavelength / 4
    pulse = response_functions.gaussian_pulse(beam, center_wavelength, bandwidth, dz)
    true_pulse = utils.discrete_gaussian(11, 2.6503)
    np.testing.assert_allclose(pulse, true_pulse, 3)

  '''Gaussian mode with mode = 0 should be identical to gaussian_axial and
  gaussian_lateral'''
  def testGaussianMode(self):
     length = 101
     wavelength = 6e-6
     dz = wavelength / 4
     axial_pulse = response_functions.gaussian_axial(length, wavelength, dz)
     axial_pulse_mode = response_functions.gaussian_axial_mode(length, wavelength, dz)
     #check absolute value as quantities are identical to a phase of kpi/4
     np.testing.assert_allclose(np.absolute(axial_pulse), np.absolute(axial_pulse_mode), 1e-10)
     lateral_pulse = response_functions.gaussian_lateral(length, wavelength, dz)
     lateral_pulse_mode = response_functions.gaussian_lateral_mode(length, wavelength, dz)
     np.testing.assert_allclose(np.absolute(lateral_pulse), np.absolute(lateral_pulse_mode), 1e-10)


if __name__ == "__main__":
  unittest.main()
