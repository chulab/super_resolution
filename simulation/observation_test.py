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
    ([2, 3, 1],),
    ([10, 12, 15, 3, 1],),
  ])
  def testIdentityPsfSingleChannel(self, state_shape):
    axial_psf = lateral_psf = tf.ones([1, 1, 1])
    state = tf.random_uniform(state_shape)
    observed_state = observation.observe(state, lateral_psf, axial_psf)
    with self.test_session() as sess:
      state_eval, observed_state_eval = sess.run([state, observed_state])
      print(state_eval.shape)
      self.assertAllClose(observed_state_eval, state_eval)

  @parameterized.expand([
    ([2, 3, 1], [1, 2.]),
    ([10, 12, 15, 3, 1], [1, 1.3, 3.4, 5.6]),
  ])
  def testIdentityPsfMultiChannel(self, state_shape, channel_multiplier):
    lateral_psf = tf.ones([1, 1, 1])
    channel_multiplier = tf.convert_to_tensor(channel_multiplier)
    axial_psf = channel_multiplier[tf.newaxis, tf.newaxis, :]
    state = tf.random_uniform(state_shape)
    true_observation = state * channel_multiplier

    observed_state = observation.observe(state, lateral_psf, axial_psf)
    with self.test_session() as sess:
      state_eval, observed_state_eval = sess.run([true_observation, observed_state])
      self.assertAllClose(observed_state_eval, state_eval)

  def testRealPsfSingleChannel(self):
    axial_psf = tf.random_uniform([4, 1, 1])
    lateral_psf = tf.random_uniform([4, 1, 1])
    state = tf.random_uniform([1, 10, 10, 1])
    observed_state = observation.observe(state, lateral_psf, axial_psf)

    kernel = axial_psf[:, tf.newaxis, :, :] * lateral_psf[tf.newaxis, :, :, :]
    real_convolution = \
      tf.nn.conv2d(state, kernel, strides=[1, 1, 1, 1],
                   padding="SAME")

    with self.test_session() as sess:
      observed_state_eval, real_eval = sess.run(
        [observed_state, real_convolution])
      self.assertAllClose(observed_state_eval, real_eval)

  def testRealPsfMultiChannelAxial(self):
    state = tf.random_uniform([1, 10, 10, 1])
    lateral_psf = tf.random_uniform([4, 1, 1])
    axial_psf = tf.random_uniform([4, 1, 2])
    observed_state = observation.observe(state, lateral_psf, axial_psf)

    kernel = axial_psf[:, tf.newaxis, :, :] * lateral_psf[tf.newaxis, :, :, :]
    real_convolution = \
      tf.nn.conv2d(state, kernel, strides=[1, 1, 1, 1],
                   padding="SAME")

    with self.test_session() as sess:
      observed_state_eval, real_eval = sess.run(
        [observed_state, real_convolution])
      self.assertAllClose(observed_state_eval, real_eval)

  def testRealPsfMultiChannelFull(self):
    state = tf.random_uniform([1, 10, 10, 1])
    lateral_psf = tf.random_uniform([4, 1, 2])
    axial_psf = tf.random_uniform([4, 2, 3])
    observed_state = observation.observe(state, lateral_psf, axial_psf)

    kernel = tf.matmul(
    tf.tile(lateral_psf[tf.newaxis, :, :, :], [4, 1, 1, 1]),
    tf.tile(axial_psf[:, tf.newaxis, :, :], [1, 4, 1, 1]),
    )

    real_convolution = \
      tf.nn.conv2d(state, kernel, strides=[1, 1, 1, 1],
                   padding="SAME")

    with self.test_session() as sess:
      observed_state_eval, real_eval = sess.run(
        [observed_state, real_convolution])
      self.assertAllClose(real_eval, observed_state_eval)


  def testComplexPsfMultiChannelFull(self):
    def rand_complex(dim):
      return np.random.randn(*dim) + 1j * np.random.randn(*dim)
    state = rand_complex([1, 2, 2, 1])
    lateral_psf = rand_complex([2, 1, 3])
    axial_psf = rand_complex([2, 3, 2])

    kernel = np.matmul(
    lateral_psf[np.newaxis, :, :, :],
    axial_psf[:, tf.newaxis, :, :],
    )

    # Perform multi-channel convolution by iterating over filters.
    real_convolution = []
    for filter in [kernel[..., i] for i in range(kernel.shape[-1])]:
      temp = np.zeros_like(state[..., 0])
      for batch in range(state.shape[-1]):
        # Need to use take conjugate of filter.
        temp[batch, :, :]= signal.correlate2d(
          state[batch, ..., 0], filter[..., 0].conj(), mode='same')
      real_convolution.append(temp)
    real_convolution = np.stack(real_convolution, -1)

    observed_state = observation.observe(
      tf.convert_to_tensor(state),
      tf.convert_to_tensor(lateral_psf),
      tf.convert_to_tensor(axial_psf)
    )

    with self.test_session() as sess:
      observed_state_eval = sess.run(observed_state)
      print(observed_state_eval[0, ..., 0])
      print(np.abs(real_convolution[0, ..., 0]))
      self.assertAllClose(np.abs(real_convolution), observed_state_eval)

  @parameterized.expand([
    ([10],),
    ([10, 10],),
  ])
  def testBadStateShape(self, shape):
    axial_psf = lateral_psf = tf.ones(shape)
    state = tf.random_uniform([10])
    with self.assertRaisesRegex(ValueError, "State must be at least 3D"):
      observation.observe(state, lateral_psf, axial_psf)


class RotateAndObserveTest(tf.test.TestCase):

  @parameterized.expand([
    ([5] * 1,),
    ([5] * 2,),
    ([5] * 4,),
  ])
  def testBadState(self, shape):
    state = tf.random_uniform(shape)
    with self.assertRaisesRegex(ValueError, "`state` must be 3D"):
      observation.rotate_and_observe(
        state, tf.ones([1]), tf.ones([1] * 3), tf.ones([1] * 3))

  def testNoRotationIdentity(self):
    state = tf.random_uniform([5, 5, 5])
    psf_lateral_filter = tf.cast(tf.ones([1] * 3), tf.complex64)
    psf_axial_filter = tf.cast(tf.ones([1] * 3), tf.complex64)
    angles = tf.zeros([7])
    observed = observation.rotate_and_observe(
      state, angles, psf_lateral_filter, psf_axial_filter
    )
    true_state = tf.tile(state[:, tf.newaxis, :, :, tf.newaxis], [1, 7, 1, 1, 1])
    with self.test_session() as sess:
      truth_eval, observed_eval = sess.run([true_state, observed])
      self.assertAllClose(truth_eval, observed_eval)

  def testWithRotationIdentity(self):
    state = tf.random_uniform([5, 5, 5])
    psf_lateral_filter = tf.cast(tf.ones([1] * 3), tf.complex64)
    psf_axial_filter = tf.cast(tf.ones([1] * 3), tf.complex64)
    angles = tf.random_uniform([6])
    observed = observation.rotate_and_observe(
      state, angles, psf_lateral_filter, psf_axial_filter
    )

    truth = []
    for slice in range(6):
      truth.append(
        tf.contrib.image.rotate(
          tf.contrib.image.rotate(state[:, :, :, tf.newaxis], angles[slice],  "BILINEAR"),
          -1 * angles[slice],  "BILINEAR",
        )
      )
    truth = tf.stack(truth, 1)

    with self.test_session() as sess:
      truth_eval, observed_eval = sess.run([truth, observed])
      self.assertAllClose(truth_eval, observed_eval)



if __name__ == "__main__":
  tf.test.main()