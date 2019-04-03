"""Tests for `signal.py`"""

import numpy as np
from scipy import signal as scipy_signal
from scipy import fftpack
import tensorflow as tf

from preprocessing import signal


class SignalTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testHilbert1D(self):
    arr = np.random.rand(24)

    hilbert_true = scipy_signal.hilbert(arr, axis=0)

    hilbert_test = signal.hilbert(tf.convert_to_tensor(arr), axis=0)

    with tf.Session() as sess:
      hilbert_test_eval = sess.run(hilbert_test)

    self.assertAllClose(
      np.abs(hilbert_test_eval), np.abs(hilbert_true), atol=1e-4)


  def testHilbert2D(self):
    arr = np.random.rand(24, 25)

    hilbert_true = scipy_signal.hilbert(arr, axis=0)

    hilbert_test = signal.hilbert(tf.convert_to_tensor(arr), axis=0)

    with tf.Session() as sess:
      hilbert_test_eval = sess.run(hilbert_test)

    self.assertAllClose(
      np.abs(hilbert_test_eval), np.abs(hilbert_true), atol=1e-4)


  def testfft(self):
    arr = np.random.rand(24, 25)
    xf = fftpack.fft(arr)
    xt = tf.fft(arr)
    with tf.Session() as sess:
      xt_eval = sess.run(xt)
    self.assertAllClose(xf, xt, atol=1e-4)