"""Test for `utils.py`"""

import numpy as np
from simulation import utils
import unittest


class utilsTest(unittest.TestCase):

  def testDiscreteGaussian(self):
    """Tests versus values computed using
    `https://keisan.casio.com/exec/system/1180573473`
    """
    sigma = 1.0
    size = 5
    real_values = 0.3678 * np.array([0.1357, 0.565, 1.266, 0.565, 0.1357])
    for real, computed in zip(real_values, utils.discrete_gaussian(size, sigma)):
      self.assertAlmostEqual(real, computed, 3)

  def testDiscreteGaussianBadSize(self):
    """Tests versus values computed using
    `https://keisan.casio.com/exec/system/1180573473`
    """
    sigma = 1.0
    size = 6
    with self.assertRaisesRegex(ValueError, "`size` must be odd."):
      utils.discrete_gaussian(size, sigma)

if __name__ == "__main__":
  unittest.main()