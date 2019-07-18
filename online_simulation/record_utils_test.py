"""Tests for `dataset_utils.py`."""

import tensorflow as tf
import numpy as np

import sys
import os
# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_simulation import record_utils


class DatasetTest(tf.test.TestCase):

  def testExampleAndParse(self):
    shape = (15, 17)
    test_prob = np.random.rand(*shape).astype(np.float32)
    test_dist = np.random.rand(*shape).astype(np.float32)
    example = record_utils.convert_to_example(test_prob, test_dist)
    example_str = example.SerializeToString()

    example = record_utils._parse_example(example_str)

    parsed_p = example['probability_distribution']
    parsed_d = example['scatterer_distribution']

    self.assertEqual(test_prob.shape, tuple(parsed_p.shape.as_list()))
    self.assertEqual(test_dist.shape, tuple(parsed_d.shape.as_list()))

    with self.test_session() as sess:
      parsed_p_, parsed_d_ = sess.run([parsed_p, parsed_d])
    self.assertAllClose(test_prob, parsed_p_)
    self.assertAllClose(test_dist, parsed_d_)
    self.assertEqual(test_prob.shape, parsed_p_.shape)
    self.assertEqual(test_dist.shape, test_dist.shape)


if __name__ == "__main__":
  tf.test.main()
