"""Tests for `dataset_utils.py`."""

import tensorflow as tf
import numpy as np
from parameterized import parameterized

from training_data import dataset_utils


class DatasetTest(tf.test.TestCase):

  @parameterized.expand([
    (1,),
    (2,),
    (4,),
  ])
  def testArrayDatasetBadShape(self, ndims):
    with self.assertRaisesRegex(ValueError, "`array` must have shape"):
      dataset_utils.array_input_fn(np.ones([5] * ndims), "EVAL", 1)

  @parameterized.expand([
    (1,),
    (2,),
    (4,),
  ])
  def testArrayDatasetEval(self, batch_size):
    array = np.random.random([20, 3, 3])
    mode = "EVAL"
    dataset = dataset_utils.array_input_fn(array, mode, batch_size)
    data = dataset.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      for batch_iter in range(20 // batch_size):
        next_batch = sess.run(data)
        self.assertAllEqual(
          array[batch_iter * batch_size:(batch_iter + 1) * batch_size],
          next_batch
        )

  @parameterized.expand([
    (1,),
    (2,),
    (4,),
  ])
  def testArrayDatasetTrain(self, batch_size):
    array = np.random.random([20, 3, 3])
    mode = "TRAIN"
    dataset = dataset_utils.array_input_fn(array, mode, batch_size)
    data = dataset.make_one_shot_iterator().get_next()
    dataset_output = []
    with self.test_session() as sess:
      for batch_iter in range(20 // batch_size):
        dataset_output.append(sess.run(data))

    for array in [array[batch_iter * batch_size:(batch_iter + 1) * batch_size]
                  for batch_iter in range(20 // batch_size)]:
      self.assertTrue(any((array == x).all() for x in dataset_output))

if __name__ == "__main__":
  tf.test.main()