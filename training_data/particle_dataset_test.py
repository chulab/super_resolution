"""Tests for `particle_dataset`."""

import numpy as np
import unittest
from parameterized import parameterized

from training_data import particle_dataset


class ParticleDatasetTest(unittest.TestCase):

  @parameterized.expand([
    (1,),
    (2,),
    (4,),
  ])
  def testBadShape(self, ndims):
    with self.assertRaisesRegex(ValueError, "`size` must be a tuple of 3"):
      particle_dataset.particle_distribution([5] * ndims)

  @parameterized.expand([
    ([100, 20, 5],),
    ([30, 20, 20],),
  ])
  def testParticleDatasetDefaultDistributionFn(self, shape):
    ds = particle_dataset.particle_distribution(shape,)

    # Check shape.
    self.assertEqual(tuple(shape), ds.shape)

    # Check bounds of distribution.
    self.assertLessEqual(np.amax(ds), 1.)
    self.assertGreaterEqual(np.amin(ds), 0.)

    # Uniform distribution should have mean approximately .5.
    self.assertAlmostEqual(np.mean(ds), .5, delta=.01)

  def testPoisson(self):
    """As `lambda_multiplier` is increased the distribution approaches input."""
    array = np.pad(
      np.ones([5, 5]), [[5, 5], [5, 5]], mode="constant")[np.newaxis, :, :]
    poisson_arrays = [particle_dataset.poisson_noise(array, multiplier)
                      for multiplier in [10, 100, 500, 1000]]
    differences = [np.sum((p[0] - array) ** 2) for p in poisson_arrays]
    for error_1, error_2 in zip(differences[:-1], differences[1:]):
      self.assertLessEqual(error_2, error_1)

  def testPoissonNormalize(self):
    """Test normalization works."""
    array = np.pad(
      np.ones([5, 5]), [[5, 5], [5, 5]], mode="constant")[np.newaxis, :, :]
    array = array * np.array([2, 6, 8])[:, np.newaxis, np.newaxis]
    poisson_array = particle_dataset.poisson_noise(array, 500, True)
    for a in poisson_array:
      self.assertAlmostEqual(1, np.amax(a))

  def testPoissonNormalizeNan(self):
    array = np.zeros([10, 5, 5])[np.newaxis, :, :]
    poisson_array = particle_dataset.poisson_noise(array, 500, True)
    for a in poisson_array:
      self.assertAlmostEqual(0, np.amax(a))

if __name__=="__main__":
  unittest.main()