"""Tests for `input.py`"""

import numpy as np

from parameterized import parameterized

import tensorflow as tf

from preprocessing import input

class testInput(tf.test.TestCase):

  @parameterized.expand([
    ((np.array([1. ,2., np.nan]), np.ones([5])), (np.zeros([3]), np.ones([5]))),
    ((np.array([1., 2., 3.]), np.array([5, np.nan])),
     (np.array([1., 2., 3.]), np.zeros([2]))),
    ((np.array([6., 5., 3.]), np.array([5., 2., 1.])),
     (np.array([6., 5., 3.]), np.array([5., 2., 1.]))),
  ])
  def testCheckForNan(self, args, expected):
    dist, obs = input.check_for_nan(*args)
    with self.test_session() as sess:
      out = sess.run([dist, obs])
      self.assertAllClose(expected, out)