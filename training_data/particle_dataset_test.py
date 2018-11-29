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



if __name__=="__main__":
  unittest.main()