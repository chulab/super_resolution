"""Tests for `numpy_convolution`."""

import unittest
import numpy as np
import tensorflow as tf

from parameterized import parameterized


from utils import numpy_convolution

class ConvolutionTest(unittest.TestCase):

  def TestPadValid(self):
    np.testing.assert_allclose([0, 0], numpy_convolution._pads(10, "VALID"))

  def testPadSame(self):
    filter_length = 4
    np.testing.assert_allclose([1, 2],
                               numpy_convolution._pads(filter_length, "SAME"))

  @parameterized.expand([
    (np.ones([1, 14, 14, 1]), np.ones([1, 1, 1, 1])),
    (np.random.rand(*[15, 32, 32, 1]),
     np.random.rand(*[5, 5, 1, 1])), # Batch dimension.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[5, 5, 4, 1])), # Multi-channel in.
    (np.random.rand(*[15, 32, 32, 3]),
     np.random.rand(*[5, 5, 3, 7])),  # Multi-channel out.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[8, 8, 4, 7])),  # Even dimensions.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[7, 7, 4, 7])),  # Odd dimensions.
    (np.random.rand(*[1, 500, 500, 1]),
     np.random.rand(*[50, 50, 1, 1])),  # Large.
  ])
  def testConvolveValid(self, tensor, filter):
    conv = numpy_convolution.convolve_2d(tensor, filter, padding="VALID")

    tf_conv = tf.nn.conv2d(tensor, filter, strides=[1]*4, padding="VALID")

    with tf.Session() as sess:
      tf_conv_eval = sess.run(tf_conv)

    np.testing.assert_allclose(tf_conv_eval, conv)

  @parameterized.expand([
    (np.ones([1, 14, 14, 1]), np.ones([1, 1, 1, 1])),
    (np.random.rand(*[15, 32, 32, 1]),
     np.random.rand(*[5, 5, 1, 1])), # Batch dimension.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[5, 5, 4, 1])), # Multi-channel in.
    (np.random.rand(*[15, 32, 32, 3]),
     np.random.rand(*[5, 5, 3, 7])),  # Multi-channel out.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[8, 8, 4, 7])),  # Even dimensions.
    (np.random.rand(*[15, 32, 32, 4]),
     np.random.rand(*[7, 7, 4, 7])),  # Odd dimensions.
    (np.random.rand(*[1, 500, 500, 1]),
     np.random.rand(*[50, 50, 1, 1])),  # Large.
  ])
  def testConvolveSame(self, tensor, filter):
    conv = numpy_convolution.convolve_2d(tensor, filter, padding="SAME")

    tf_conv = tf.nn.conv2d(tensor, filter, strides=[1]*4, padding="SAME")

    with tf.Session() as sess:
      tf_conv_eval = sess.run(tf_conv)

    np.testing.assert_allclose(tf_conv_eval, conv)


if __name__ == "__main__":
  unittest.main()