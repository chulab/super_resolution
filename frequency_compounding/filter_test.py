"""Tests for `filter.py`."""

import unittest
import random
import numpy as np

from frequency_compounding import filter
from frequency_compounding import utils

class filterTest(unittest.TestCase):

  def testGaussianFilter(self):
    index = np.arange(0, 5, .01)
    frequency = [random.random() * 5. for _ in range(3)]
    sigma = [random.random() * 2. for _ in range(3)]
    filters = filter.gaussian_filter(index, frequency, sigma)

    np.testing.assert_equal(filters.shape, (len(index), 3))

    for f in range(3):
      np.testing.assert_allclose(
        filters[:, f],
        utils._gaussian(index, frequency[f], sigma[f]) +
        utils._gaussian(index, - frequency[f], sigma[f])
      )

  def testGaussianFilterBadArgs(self):
    index = np.arange(0, 5, .01)
    frequency = [random.random() * 5. for _ in range(3)]
    sigma = [random.random() * 2. for _ in range(4)]
    with self.assertRaisesRegex(ValueError, "`Frequency` and `sigma` should "
                                            "have same length."):
      filter.gaussian_filter(index, frequency, sigma)

  def testExtractFrequencyBadBatch(self):
    signal = np.random.rand(12, 3, 5)
    filters = np.random.rand(13, 3, 5, 1)
    with self.assertRaisesRegex(ValueError, "`signal` and `filters` must have "
                                            "compatible batch"):
      filter.extract_frequency(signal, filters)

  def testExtractFrequencyBadFilterLength(self):
    signal = np.random.rand(12, 3, 5)
    filters = np.random.rand(12, 3, 6, 1)
    with self.assertRaisesRegex(ValueError, "`signal` and `filters` must have "
                                            "same `filter_length`."):
      filter.extract_frequency(signal, filters)

  def testExtractFrequency(self):
    sample_frequency = .02
    sample_length = 5
    t = np.arange(0, sample_length, sample_frequency)
    signal_1 = np.sin(t * np.pi * 2 * 1)
    signal_2 = np.sin(t * np.pi * 2 * 3)
    signal = signal_1 + signal_2
    index = np.fft.fftshift(np.fft.fftfreq(signal.shape[0], sample_frequency))

    filters = filter.gaussian_filter(index, [1., 3., 7], [.1, .1, .1])

    extracted = filter.extract_frequency(signal, filters)

    np.testing.assert_allclose(np.real(extracted[:, 0]), signal_1, atol=.001)
    np.testing.assert_allclose(np.real(extracted[:, 1]), signal_2, atol=.001)
    np.testing.assert_allclose(
      np.real(extracted[:, 2]), np.zeros_like(signal_1), atol=.001)

if __name__ == "__main__":
  unittest.main()
