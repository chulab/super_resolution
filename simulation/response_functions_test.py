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

if __name__ == "__main__":
  unittest.main()