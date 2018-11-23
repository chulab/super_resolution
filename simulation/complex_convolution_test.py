"""Tests for `complex_convolution`"""

import numpy as np
import tensorflow as tf
from parameterized import parameterized

from simulation import complex_convolution


class ComplexConvolutionTest(tf.test.TestCase):

  @parameterized.expand([
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME"),
  ])
  def testConvolutionReal(self, stride, padding):
    tensor = tf.random_uniform([10])
    filter = tf.random_uniform([3, 4])
    convolved = complex_convolution.convolve_complex_1d(tensor, filter, stride,
                                                        padding)
    normal_convolve_1d = tf.nn.conv1d(
      tensor[tf.newaxis, :, tf.newaxis], filter[:, tf.newaxis, :],
      stride, padding)[0, :, :]
    with self.test_session() as sess:
      convolved_eval, true_value = sess.run([convolved, normal_convolve_1d])
      self.assertAllClose(convolved_eval, true_value)

  @parameterized.expand([
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME"),
  ])
  def testConvolutionNDwith1D(self, stride, padding):
    tensor = tf.random_uniform([2, 10])
    filter = tf.random_uniform([3, 4])
    convolved = complex_convolution.convolve_complex_1d(tensor, filter, stride,
                                                        padding)
    normal_convolve_1d = tf.nn.conv1d(
      tensor[..., tf.newaxis], filter[:, tf.newaxis, :],
      stride, padding)
    with self.test_session() as sess:
      convolved_eval, true_value = sess.run([convolved, normal_convolve_1d])
      self.assertAllClose(convolved_eval, true_value)

  @parameterized.expand([
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME"),
  ])
  def testConvolutionNDwith1DComplex(self, stride, padding):
    tensor = tf.random_uniform([2, 10])
    filter = tf.random_uniform([3, 3])
    convolved = complex_convolution.convolve_complex_1d(
      tf.complex(real=tf.zeros_like(tensor), imag=tensor),
      tf.cast(filter, tf.complex64), stride,
      padding)
    normal_convolve_1d = tf.nn.conv1d(
      tensor[..., tf.newaxis], filter[:, tf.newaxis, :],
      stride, padding)
    with self.test_session() as sess:
      convolved_eval, true_value = sess.run([convolved, normal_convolve_1d])
      self.assertAllClose(np.abs(convolved_eval), true_value)

  def testConvolutionComplex(self):
    tensor = tf.constant([1 + 2j, 0., 1j, 1 + 3j], dtype=tf.complex64)
    filter = tf.constant([[0.+2.j, 0.+1.j], [3.+0.j, 2.+1.j]])
    convolved = complex_convolution.convolve_complex_1d(tensor,
                                                        tf.cast(filter,
                                                                tf.complex64))
    truth_value = [[-4.+2.j, -2.+1.j], [ 0.+3.j, -1.+2.j], [ 1.+9.j, -2.+7.j]]
    with self.test_session() as sess:
      convolved_eval = sess.run(convolved)
      self.assertAllClose(np.real(convolved_eval), np.real(truth_value))
      self.assertAllClose(np.abs(convolved_eval), np.abs(truth_value))

  def testInvalidPadding(self):
    tensor = tf.constant([1., 2.])
    filter = tf.constant([[1., 2.]])
    with self.assertRaisesRegex(ValueError, "`padding` must be one of"):
      complex_convolution.convolve_complex_1d(tensor, filter,
                                              padding="NotValid")

  def testInvalidKernel(self):
    tensor = tf.constant([1., 2.])
    filter = tf.constant([1., 2.])
    with self.assertRaisesRegex(ValueError,
                                "Shapes \(2,\) and \[None, None\]"):
      complex_convolution.convolve_complex_1d(tensor, filter, )

  def testInvalidDtypes(self):
    tensor = tf.constant([1., 2.], dtype=tf.float32)
    filter = tf.constant([[1., 2.]], dtype=tf.float64)
    with self.assertRaisesRegex(ValueError, "`tensor` and `filter` must"):
      complex_convolution.convolve_complex_1d(tensor, filter, )


if __name__ == "__main__":
  tf.test.main()
