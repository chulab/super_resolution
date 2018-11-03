"""Tests for `observation.py`."""

import tensorflow as tf
from parameterized import parameterized
from scipy import signal
import numpy as np

from simulation import observation


class ObservationTest(tf.test.TestCase):

  def startUp(self):
    tf.reset_default_graph()

  @parameterized.expand([
    ([2, 3, 4],),
    ([10, 12, 15, 3, 4],),
  ])
  def testIdentityPsf(self, state_shape):
    axial_psf = lateral_psf = tf.ones([1])
    state = tf.random_uniform(state_shape)
    observed_state = observation.observe(state, lateral_psf, axial_psf)
    with self.test_session() as sess:
      state_eval, observed_state_eval = sess.run([state, observed_state])
      self.assertAllClose(observed_state_eval, state_eval)

  def testRealPsf(self):
    axial_psf = tf.random_uniform([4])
    lateral_psf = tf.random_uniform([4])
    state = tf.random_uniform([10, 10, 10])
    observed_state = observation.observe(state, lateral_psf, axial_psf)

    kernel = axial_psf[:, tf.newaxis] * lateral_psf[tf.newaxis, :]
    kernel = kernel[..., tf.newaxis, tf.newaxis]
    real_convolution = \
      tf.nn.conv2d(state[..., tf.newaxis], kernel, strides=[1, 1, 1, 1],
                   padding="SAME")[..., 0]

    with self.test_session() as sess:
      observed_state_eval, real_eval = sess.run(
        [observed_state, real_convolution])
      self.assertAllClose(observed_state_eval, real_eval)

  def testComplexPsf(self):
    axial_psf = np.random.rand(*[4]) + 1j * np.random.rand(*[4])
    lateral_psf = np.random.rand(*[4]) + 1j * np.random.rand(*[4])
    state = np.random.rand(*[10, 10])
    observed_state = observation.observe(
      tf.constant(state, dtype=tf.complex64),
      tf.constant(lateral_psf, dtype=tf.complex64),
      tf.constant(axial_psf, dtype=tf.complex64))

    kernel = axial_psf[:, tf.newaxis] * lateral_psf[tf.newaxis, :]
    real_convolution = signal.correlate2d(state, kernel, mode="same")
    real_convolution = np.abs(real_convolution)
    with self.test_session() as sess:
      observed_state_eval = sess.run(observed_state)
      self.assertAllClose(observed_state_eval, real_convolution)

  def testBadStateShape(self):
    axial_psf = lateral_psf = tf.ones([1])
    state = tf.random_uniform([10])
    with self.assertRaisesRegex(ValueError, "State must be at least 2D"):
      observation.observe(state, lateral_psf, axial_psf)

  def testBadStateAxialPSF(self):
    lateral_psf = tf.ones([1])
    axial_psf = tf.ones([10, 1])
    state = tf.random_uniform([10, 10])
    with self.assertRaisesRegex(ValueError, "Both PSF's must be 1D"):
      observation.observe(state, lateral_psf, axial_psf)

  def testBadStateLateralPSF(self):
    lateral_psf = tf.ones([10, 1])
    axial_psf = tf.ones([1])
    state = tf.random_uniform([10, 10])
    with self.assertRaisesRegex(ValueError, "Both PSF's must be 1D"):
      observation.observe(state, lateral_psf, axial_psf)
