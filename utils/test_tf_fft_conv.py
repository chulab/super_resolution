"""Tests for `tf_fft_conv.py`"""

import tensorflow as tf

import scipy.signal as signal
import numpy as np

import sys
import os
# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

  def test_fftshift_split_even(self):
    a = tf.random.uniform([50, 50])

    v1 = tf_fft_conv.fftshift(a)
    v2 = tf_fft_conv.fftshift_split(a)

    with self.test_session() as sess:
      v1_, v2_ = sess.run([v1, v2])
    self.assertAllClose(v1_, v2_, atol=.0001)

  def test_fftshift_split_odd(self):
    a = tf.random.uniform([55, 51])

    v1 = tf_fft_conv.fftshift(a)
    v2 = tf_fft_conv.fftshift_split(a)

    with self.test_session() as sess:
      v1_, v2_ = sess.run([v1, v2])
    self.assertAllClose(v1_, v2_, atol=.0001)

  def test_ifftshift_split_even(self):
    a = tf.random.uniform([50, 50])

    v1 = tf_fft_conv.ifftshift(a)
    v2 = tf_fft_conv.ifftshift_split(a)

    with self.test_session() as sess:
      v1_, v2_ = sess.run([v1, v2])
    self.assertAllClose(v1_, v2_, atol=.0001)

  def test_ifftshift_split_odd(self):
    a = tf.random.uniform([55, 51])

    v1 = tf_fft_conv.ifftshift(a)
    v2 = tf_fft_conv.ifftshift_split(a)

    with self.test_session() as sess:
      v1_, v2_ = sess.run([v1, v2])
    self.assertAllClose(v1_, v2_, atol=.0001)

if __name__ == "__main__":
  tf.test.main()
