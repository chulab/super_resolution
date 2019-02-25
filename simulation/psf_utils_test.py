"""Tests for `psf_utils`."""

import unittest
import numpy as np

from simulation import psf_utils
from simulation import defs_test


class PsfUtilsTest(unittest.TestCase):

  def testSameMode(self):
    psfs = [np.random.rand(10) for _ in range(5)]
    filter = psf_utils.to_filter(psfs, 'FROM_SAME')
    self.assertEqual(filter.shape, (10, 5, 5))
    for in_channel_iter in range(5):
      for filter_iter in range(5):
        if in_channel_iter == filter_iter:
          np.testing.assert_allclose(
            filter[:, in_channel_iter, filter_iter], psfs[in_channel_iter])
        else:
          np.testing.assert_equal(
            filter[:, in_channel_iter, filter_iter], np.zeros_like(psfs[0]))

  def testSameMode2D(self):
    psfs = [np.random.rand(10, 10) for _ in range(5)]
    filter = psf_utils.to_filter(psfs, 'FROM_SAME')
    self.assertEqual(filter.shape, (10, 10, 5, 5))
    for in_channel_iter in range(5):
      for filter_iter in range(5):
        if in_channel_iter == filter_iter:
          np.testing.assert_allclose(
            filter[:, :, in_channel_iter, filter_iter], psfs[in_channel_iter])
        else:
          np.testing.assert_equal(
            filter[:, :, in_channel_iter, filter_iter], np.zeros_like(psfs[0]))

  def testFromSingleMode(self):
    psfs = [np.random.rand(10) for _ in range(5)]
    filter = psf_utils.to_filter(psfs, 'FROM_SINGLE')
    self.assertEqual(filter.shape, (10, 1, 5))
    for filter_iter in range(5):
      np.testing.assert_allclose(filter[:, 0, filter_iter], psfs[filter_iter])

  def testFromSingleMode2D(self):
    psfs = [np.random.rand(10, 10) for _ in range(5)]
    filter = psf_utils.to_filter(psfs, 'FROM_SINGLE')
    self.assertEqual(filter.shape, (10, 10, 1, 5))
    for filter_iter in range(5):
      np.testing.assert_allclose(filter[:, :, 0, filter_iter], psfs[filter_iter])


  def testBadMode(self):
    mode = "NOT_A_MODE"
    with self.assertRaisesRegex(ValueError, "`mode` must be one of"):
      psf_utils.to_filter([np.ones(1)], mode)

  def testIncompatiblePSFLength(self):
    psfs = [np.ones(12), np.ones(5)]
    with self.assertRaisesRegex(ValueError, "All PSF's must have same shape"):
      psf_utils.to_filter(psfs, "FROM_SAME")

  def testBadLateralShape(self):
    with self.assertRaisesRegex(
        ValueError, "`psf_length` must be odd."):
      psf_utils.lateral_psf_filters(12, [.1, .2], 2., .1)

  def testBadAxialShape(self):
    with self.assertRaisesRegex(
        ValueError, "`psf_length` must be odd."):
      psf_utils.axial_psf_filters(12, [.1, .2], 2., .1, .1)

  def testFromObservationSpec(self):
    pass

  def testFromObservationSpecBadType(self):
    with self.assertRaisesRegex(ValueError, "`type` must be one of"):
      psf_utils.psf_filter("NOT_A_REAL_TYPE", 11,
                           defs_test.simple_observation_spec())


if __name__ == "__main__":
  unittest.main()
