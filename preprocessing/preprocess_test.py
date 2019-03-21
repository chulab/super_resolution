"""Tests for `preprocess.py`"""

import numpy as np

import tensorflow as tf

from preprocessing import preprocess


class preprocessTest(tf.test.TestCase):

  def test_gaussian_kernel(self):
    pass


class imageBlurTest(tf.test.TestCase):

  def test_image_blur(self):
    grid_pitch = .1
    sigma_blur = .5
    kernel_size = 1.
    image_blur = preprocess.imageBlur(grid_pitch, sigma_blur, kernel_size)
    self.assertAllClose((21, 21, 1, 1), image_blur._kernel.shape)
    self.assertAllClose(
      preprocess._gaussian_kernel(10, 5.),
      image_blur._kernel[..., 0, 0]
    )

  def test_image_blur_multi_channel(self):
    grid_pitch = .1
    sigma_blur = .5
    kernel_size = 1.
    blur_channels = 5
    image_blur = preprocess.imageBlur(
      grid_pitch, sigma_blur, kernel_size, blur_channels)
    self.assertAllClose((21, 21, blur_channels, blur_channels),
                        image_blur._kernel.shape)
    with self.test_session():
      for channel_in in range(blur_channels):
        for channel_out in range(blur_channels):
          if channel_in == channel_out:
            self.assertAllClose(
              preprocess._gaussian_kernel(10, 5.),
              image_blur._kernel[..., channel_in, channel_out]
            )
          else:
            self.assertAllClose(
              np.zeros([21, 21]),
              image_blur._kernel[..., channel_in, channel_out]
            )


  def test_image_blur_low_sigma(self):
    grid_pitch = 1.
    sigma_blur = .01
    kernel_size = 1.
    blur_channels = 5
    image_blur = preprocess.imageBlur(
      grid_pitch, sigma_blur, kernel_size, blur_channels)

    test_image = tf.random.uniform([1, 10, 10, blur_channels])
    blurred_image = image_blur.blur(test_image)

    with self.test_session() as sess:
      test_image_eval, blurred_image_eval = sess.run([test_image, blurred_image])
      self.assertAllClose(test_image_eval, blurred_image_eval)

  def test_image_blur_low_sigma_multi_dimension(self):
    grid_pitch = 1.
    sigma_blur = .01
    kernel_size = 1.

    image_blur = preprocess.imageBlur(grid_pitch, sigma_blur, kernel_size)

    test_image = tf.random.uniform([1, 10, 10, 1])
    blurred_image = image_blur.blur(test_image)

    with self.test_session() as sess:
      test_image_eval, blurred_image_eval = sess.run([test_image, blurred_image])
      self.assertAllClose(test_image_eval, blurred_image_eval)

if __name__ == "__main__":
  tf.test.main()