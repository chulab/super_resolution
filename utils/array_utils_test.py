"""Tests for `array_utils.py`."""

import unittest
from parameterized import parameterized

import numpy as np

from utils import array_utils

class arrayUtilsTest(unittest.TestCase):


  @parameterized.expand([
    ((5, 7, 9), (5, 7, 9), True),
    ((5, 7, 9), (5, 7, 1), True),
    ((5, 7, 9), (5, 1, 9), True),
    ((5, 7, 9), (1, 1, 9), True),
    ((12, 7, 9), (6, 1, 9), False),
    ((12, 7), (12, 1, 7), False),
    ((12,), (5,), False),
  ])
  def testIsBroadCastCompatible(self, shape_a, shape_b, compatible):
    self.assertTrue(
      compatible == array_utils.is_broadcast_compatible(shape_a, shape_b)
    )

  @parameterized.expand([
    ((41, 23, 4), 0),
    ((12, 14), 1),
    ((23, 41, 33), 1),
  ])
  def testArraySplit(self, shape, axis):
    array = np.random.rand(*shape)
    splits = array_utils.reduce_split(array, axis)
    for i in range(array.shape[axis]):
      np.testing.assert_equal(splits[i], array.take(i, axis))