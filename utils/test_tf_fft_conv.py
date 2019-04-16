"""Tests for `tf_fft_conv.py`"""

import tensorflow as tf

import scipy.signal as signal
import numpy as np


from utils import tf_fft_conv


class testTFFFTConv(tf.test.TestCase):

  def testConv2d(self):
    a = np.random.rand(50, 50)
    b = np.random.rand(20, 20)
    mode='same'
    scipy_result = signal.fftconvolve(a, b, mode=mode)
    tf_result = tf_fft_conv.fft_conv(
      tf.convert_to_tensor(a), tf.convert_to_tensor(b), mode=mode)
    with self.test_session() as sess:
      tf_result = sess.run(tf_result)
    self.assertAllClose(scipy_result, tf_result, atol=.0001)

  def testCorrelate(self):
    a = np.random.rand(50, 50)
    b = np.random.rand(20, 20)
    mode='same'
    scipy_result = signal.correlate(a, b, method='fft', mode=mode)
    tf_result = tf_fft_conv.fft_correlate(
      tf.convert_to_tensor(a), tf.convert_to_tensor(b), mode=mode)
    with self.test_session() as sess:
      tf_result = sess.run(tf_result)
    self.assertAllClose(scipy_result, tf_result, atol=.0001)