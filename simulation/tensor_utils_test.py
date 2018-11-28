"""Tests for `tensor_utils`."""

import tensorflow as tf
from parameterized import parameterized

from simulation import tensor_utils


class RotateTest(tf.test.TestCase):

  @parameterized.expand([
    ([5],),
    ([5, 5],),
    ([5, 5, 5],),
    ])
  def testBadShape(self, shape):
    tensor = tf.ones(shape)
    angles = tf.convert_to_tensor([1])
    rotation_axis = 1
    with self.assertRaisesRegex(ValueError, "`tensor` must have rank at"):
      tensor_utils.rotate_tensor(tensor, angles, rotation_axis)

  def testNegativeRotationAxis(self):
    tensor = tf.ones([5] * 4)
    angles = tf.convert_to_tensor([1])
    rotation_axis = -2
    with self.assertRaisesRegex(
        ValueError, "`rotation_axis` must be positive."):
      tensor_utils.rotate_tensor(tensor, angles, rotation_axis)

  @parameterized.expand([
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 2),
    (5, 7),
  ])
  def testInvalidRotationAxis(self, dimensions, rotation_axis):
    tensor = tf.ones([5] * dimensions)
    angles = tf.convert_to_tensor([1])
    with self.assertRaisesRegex(
        ValueError, "`rotation_axis` must be a batch dimension."):
      tensor_utils.rotate_tensor(tensor, angles, rotation_axis)

  def testInvalidangleshape(self):
    tensor = tf.ones([5] * 4)
    angles = tf.convert_to_tensor([[1]] * 3)
    rotation_axis = 0
    with self.assertRaisesRegex(
        ValueError, "`angles` must be a 1D list."):
      tensor_utils.rotate_tensor(tensor, angles, rotation_axis)

  def testIncompatibleDimension(self):
    tensor = tf.ones([5] * 4)
    angles = tf.convert_to_tensor([1])
    rotation_axis = 0
    with self.assertRaisesRegex(
        ValueError, "`angles` length must equal `rotation_axis`"):
      tensor_utils.rotate_tensor(tensor, angles, rotation_axis)

  def testRotationNoBatch(self):
    tensor = tf.random_uniform([5] * 4)
    angles = tf.random_uniform([5])
    rotation_axis = 0
    true_rotation = tf.contrib.image.rotate(tensor, angles, "BILINEAR")
    rotate = tensor_utils.rotate_tensor(tensor, angles, rotation_axis)
    with self.test_session() as sess:
      truth, rotate_eval = sess.run([true_rotation, rotate])
      self.assertAllClose(truth, rotate_eval)

  def testRotationBatch(self):
    tensor = tf.random_uniform([3, 7, 5, 5, 5])
    angles = tf.random_uniform([7])
    rotation_axis = 1
    true_rotations = []
    for batch_iter in range(3):
      true_rotations.append(tf.contrib.image.rotate(
        tensor[batch_iter], angles, "BILINEAR"))
    true_rotation = tf.stack(true_rotations, 0)
    rotate = tensor_utils.rotate_tensor(tensor, angles, rotation_axis)
    with self.test_session() as sess:
      truth, rotate_eval = sess.run([true_rotation, rotate])
      self.assertAllClose(truth, rotate_eval)


class TransposeTest(tf.test.TestCase):

  @parameterized.expand([
    ([0, 1], [0, 1],),
    ([1, 0], [1, 0],),
    ([2, 5, 3, 4, 0, 1], [4, 5, 0, 2, 3, 1],),
  ])
  def testReverseTranspose(self, sequence, reverse):
    print(sequence)
    self.assertEqual(
      tensor_utils._reverse_transpose_sequence(sequence), reverse)

  def testCombineBatchIntoChannels(self):
    tensor = tf.random_uniform([5, 10, 7, 8, 9])

    combined = tensor_utils.combine_batch_into_channels(
      tensor, 0
    )

    true_combination = tf.reshape(
      tf.transpose(tensor, [0, 2, 3, 4, 1]),
      [5, 7, 8, 9 * 10]
    )

    with self.test_session() as sess:
      combined_eval, truth_eval = sess.run([combined, true_combination])

    self.assertAllEqual(truth_eval, combined_eval)

  @parameterized.expand([
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (5, 2),
    (5, 7),
  ])
  def testCombineBatchBadExcludeDimension(self, dimensions, exclude_dimension):
    tensor = tf.ones([5] * dimensions)
    with self.assertRaisesRegex(
        ValueError, "`exclude_dimension` must be a batch dimension."):
      tensor_utils.combine_batch_into_channels(tensor, exclude_dimension)


if __name__ == "__main__":
  tf.test.main()