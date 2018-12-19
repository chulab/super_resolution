"""Tests for `estimator.py`"""

import numpy as np
import tensorflow as tf

from simulation import estimator
from simulation import psf_utils
from simulation import observation
from simulation import defs_test


class ObservationEstimatorTest(tf.test.TestCase):

  def testMakeEstimator(self):
    estimator.SimulationEstimator(
      defs_test.simple_observation_spec(), 3, 3)

  def testObservationSimulation(self):
    distribution = np.random.rand(10, 20, 20)

    # Problem params.
    angles = [0.]
    frequencies = [6e6]
    grid_dimension = 5e-4
    transducer_bandwidth = 1.
    numerical_aperture = 1.
    lateral_length = 3
    axial_length = 3

    observation_spec = defs_test.simple_observation_spec(
      angles, frequencies, grid_dimension, transducer_bandwidth,
      numerical_aperture)

    # Set up using normal graph.
    test_state = tf.convert_to_tensor(distribution)

    lateral_psf = psf_utils.psf_filter(psf_utils._LATERAL, lateral_length,
                                       observation_spec)
    axial_psf = psf_utils.psf_filter(psf_utils._AXIAL, axial_length,
                                     observation_spec)

    test_psf_lateral_filter = tf.convert_to_tensor(lateral_psf,
                                                   dtype=tf.complex64)
    test_psf_axial_filter = tf.convert_to_tensor(axial_psf,
                                                 dtype=tf.complex64)

    test_angles = tf.convert_to_tensor(angles)

    new_simulated_observation = observation.rotate_and_observe(
      test_state, test_angles, test_psf_lateral_filter, test_psf_axial_filter)

    with tf.Session() as sess:
      real_result = sess.run(new_simulated_observation)

    # Run using estimator.
    test_estimator = estimator.SimulationEstimator(
      observation_spec, lateral_length, axial_length)

    def input_fn():
      return tf.data.Dataset.from_tensor_slices((distribution)).batch(
        distribution.shape[0])

    estimator_result = test_estimator.predict(input_fn,
                                              yield_single_examples=False)

    self.assertAllClose(real_result, next(estimator_result)["observation"])

  def testSimulationEstimatorBadMode(self):
    distribution = np.random.rand(10, 20, 20)
    # Run using estimator.
    test_estimator = estimator.SimulationEstimator(
      defs_test.simple_observation_spec(), 5, 5)

    def input_fn():
      return tf.data.Dataset.from_tensor_slices((distribution)).batch(
        distribution.shape[0])

    with self.assertRaisesRegex(ValueError,
                                "Only `PREDICT` mode is supported"):
      test_estimator.train(input_fn, steps=20)


if __name__ == "__main__":
  tf.test.main()
