"""Tests for `utils.py`"""

import unittest
import numpy as np

from frequency import utils


class utilsTest(unittest.TestCase):

  def testPad(self):
    test_array = np.random.rand(10, 12)
    padded = utils._pad(test_array, 7)
    np.testing.assert_equal(padded.shape, (10, 19))
    np.testing.assert_equal(padded[:, -7:], np.zeros((10, 7)))

  def testBadSampleShape(self):
    samples = np.array(1)
    sample_frequency = .01
    with self.assertRaisesRegex(ValueError, "`samples` must be at least 1D"):
      utils._fft(samples, sample_frequency)

  def testBadFrequency(self):
    samples = np.random.rand(12, 2)
    sample_frequency = 3
    with self.assertRaisesRegex(ValueError, "`sample_frequency` must be"):
      utils._fft(samples, sample_frequency)

  def testFFT(self):
    samples = np.sin(np.linspace(0, np.pi * 2, 20))
    sample_frequency = 1. / 20.
    fft, ind = utils._fft(samples, sample_frequency)
    true_fft = np.array(
      [-1.66870486e-01 - 2.77555756e-17j, -1.66987131e-01 - 2.64481634e-02j,
       -1.67362491e-01 - 5.43793697e-02j, -1.68085612e-01 - 8.56438972e-02j,
       -1.69359869e-01 - 1.23047147e-01j, -1.71650217e-01 - 1.71650217e-01j,
       -1.76163417e-01 - 2.42468142e-01j, -1.86918926e-01 - 3.66849048e-01j,
       -2.26651931e-01 - 6.97562918e-01j, 1.51661484e+00 + 9.57552923e+00j,
       1.11022302e-16 + 0.00000000e+00j, 1.51661484e+00 - 9.57552923e+00j,
       -2.26651931e-01 + 6.97562918e-01j, -1.86918926e-01 + 3.66849048e-01j,
       -1.76163417e-01 + 2.42468142e-01j, -1.71650217e-01 + 1.71650217e-01j,
       -1.69359869e-01 + 1.23047147e-01j, -1.68085612e-01 + 8.56438972e-02j,
       -1.67362491e-01 + 5.43793697e-02j, -1.66987131e-01 + 2.64481634e-02j])
    true_ind = [-10., -9., -8., -7., -6., -5., -4., -3., -2., -1., 0.,
                1., 2., 3., 4., 5., 6., 7., 8., 9.]
    np.testing.assert_allclose(np.abs(true_fft), np.abs(fft), atol=1e-7)
    np.testing.assert_allclose(true_ind, ind)

  def testGaussian(self):
    coordinates = np.linspace(0, 10, 20)
    gaussian = utils._gaussian(coordinates, 2., 1.5)
    true_gaussian = [
      4.11112291e-01, 6.17170452e-01, 8.19184469e-01,
      9.61369222e-01, 9.97540733e-01, 9.15172544e-01,
      7.42347492e-01,
      5.32406598e-01, 3.37607069e-01, 1.89282976e-01,
      9.38303848e-02,
      4.11251372e-02, 1.59368741e-02, 5.46048029e-03,
      1.65420933e-03,
      4.43079856e-04, 1.04931400e-04, 2.19715577e-05,
      4.06769281e-06,
      6.65836147e-07]
    np.testing.assert_allclose(true_gaussian, gaussian)


if __name__ == "__main__":
  unittest.main()
