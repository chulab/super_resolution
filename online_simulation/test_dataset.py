"""Tests for `dataset_utils.py`."""

import tensorflow as tf
import numpy as np
from parameterized import parameterized

import sys
import os
# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_simulation import dataset


class DatasetTest(tf.test.TestCase):

  def testExampleAndParse(self):
    shape = (15, 17)
    test_prob = np.random.rand(*shape).astype(np.float32)
    test_dist = np.random.rand(*shape).astype(np.float32)
    example = dataset.convert_to_example(test_prob, test_dist)
    example_str = example.SerializeToString()

    parsed_p, parsed_d = dataset._parse_example(example_str)

    self.assertEqual(test_prob.shape, tuple(parsed_p.shape.as_list()))
    self.assertEqual(test_dist.shape, tuple(parsed_d.shape.as_list()))

    with self.test_session() as sess:
      parsed_p_, parsed_d_ = sess.run([parsed_p, parsed_d])
    self.assertAllClose(test_prob, parsed_p_)
    self.assertAllClose(test_dist, parsed_d_)
    self.assertEqual(test_prob.shape, parsed_p_.shape)
    self.assertEqual(test_dist.shape, test_dist.shape)


    # @parameterized.expand([
    #   (1,),
    #   (2,),
    #   (4,),
    # ])
    # def testExampleAndParse(self, ndims):
    #
    #
    #   with self.assertRaisesRegex(ValueError, "`array` must have shape"):
    #     dataset_utils.array_input_fn(np.ones([5] * ndims), "EVAL", 1)



if __name__ == "__main__":
  tf.test.main()
