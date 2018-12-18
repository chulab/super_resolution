"""Tests for `generate.py`"""

import numpy as np
import tensorflow as tf
from parameterized import parameterized

from training_data import generate
from simulation import observation_estimator


class GenerateTest(tf.test.TestCase):

  @parameterized.expand([
    ([100, 20, 5], [12, 4, 5]),
    ([30, 20, 20], [1, 30]),
    ([30], [2, 50, 32, 4]),
  ])
  def testEncodeAndDecodeExample(
      self, distribution_shape, observation_shape):
    # Makes and reloads example array.

    true_scatter = np.random.rand(*distribution_shape).astype(np.float32)
    true_observation = np.random.rand(*observation_shape).astype(np.float32)

    true_observation_spec = observation_estimator.ObservationSpec([0, np.pi / 2], [1.5], .23, .2, .8)
    example = generate._construct_example(
      true_scatter, true_observation, true_observation_spec)

    example_str = example.SerializeToString()

    scatterer_distribution_, observation_, observation_params_ = generate._parse_example(
      example_str)
    with tf.Session() as sess:
      test_scatter, test_observation, test_observation_params = sess.run(
        [scatterer_distribution_, observation_, observation_params_])

    self.assertAllEqual(true_scatter, test_scatter)
    self.assertAllClose(true_observation, test_observation)
    self.assertTrue(observation_params_, true_observation_spec._asdict())

if __name__=="__main__":
  tf.test.main()