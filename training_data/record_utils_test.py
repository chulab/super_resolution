"""Tests for `record_utils.py`"""

import numpy as np
import tensorflow as tf
from parameterized import parameterized

from training_data import record_utils
from simulation import defs


class GenerateTest(tf.test.TestCase):

  @parameterized.expand([
    ([100, 20, 5], [12, 4, 5]),
    ([30, 20, 20], [1, 30, 25, 7]),
    ([30, 25], [2, 50, 32, 4]),
  ])
  def testEncodeAndDecodeExample(
      self, distribution_shape, observation_shape):
    # Makes and reloads example array.

    true_scatter = np.random.rand(*distribution_shape).astype(np.float32)
    true_observation = np.random.rand(*observation_shape).astype(np.float32)

    example = record_utils._construct_example(
      true_scatter, true_observation)

    example_str = example.SerializeToString()

    scatterer_distribution_, observation_ = record_utils._parse_example(
      example_str)
    with tf.Session() as sess:
      test_scatter, test_observation = sess.run(
        [scatterer_distribution_, observation_])
    self.assertAllEqual(true_scatter, test_scatter)
    self.assertAllClose(true_observation, test_observation)

  def testEncodeAndDecodeWithShape(self):
    true_scatter = np.random.rand(24, 46).astype(np.float32)
    true_observation = np.random.rand(12, 35, 46, 3).astype(np.float32)

    example = record_utils._construct_example(
      true_scatter, true_observation)

    example_str = example.SerializeToString()

    scatterer_distribution_, observation_, sd_shape_, ob_shape_ = record_utils._parse_example(
      example_str, True)
    with tf.Session() as sess:
      test_scatter, test_observation, sd_shape, ob_shape = sess.run(
        [scatterer_distribution_, observation_, sd_shape_, ob_shape_])
    self.assertAllEqual(true_scatter, test_scatter)
    self.assertAllClose(true_observation, test_observation)
    self.assertAllClose([24, 46], sd_shape)
    self.assertAllClose([35, 46], ob_shape)

  def testBadDtypeDistribution(self):
    true_scatter = np.random.rand(400, 30, 2).astype(np.double)
    true_observation = np.random.rand(400, 30, 2).astype(np.float32)
    with self.assertRaisesRegex(ValueError, "`distribution` must have dtype"):
      record_utils._construct_example(true_scatter, true_observation)

  def testBadDtypeObservation(self):
    true_scatter = np.random.rand(400, 30, 2).astype(np.float32)
    true_observation = np.random.rand(400, 30, 2).astype(np.double)
    with self.assertRaisesRegex(ValueError, "`observation` must have dtype"):
      record_utils._construct_example(true_scatter, true_observation)


if __name__=="__main__":
  tf.test.main()