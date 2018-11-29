"""Methods to simulate US image formation."""

import tensorflow as tf
from typing import Optional, Tuple

from simulation import complex_convolution
from simulation import tensor_utils


def observe(
    state: tf.Tensor,
    psf_lateral: tf.Tensor,
    psf_axial: tf.Tensor,
) -> tf.Tensor:
  """Simulates an observation of `state` using the convolution method.

  This function simulates and US observation by a device which is parameterized
  by an axial and lateral PSF. This method is verified in:
    `Gao et al., Jan. (2009). A Fast Convolution-Based Methodology to
    Simulate 2-D/3-D Cardiac Ultrasound Images.`

  This function first convolves `state` with `psf_lateral` and then
  `psf_axial`.  Explicitly:
    `psf_axial X psf_lateral X state`
  where `X` represents convolution.

  If `psf_lateral` and `psf_axial` are complex, then this function generates
  the RF signal generated by imaging state including any interference effects.

  Args:
    state: `tf.Tensor` of shape `batch_dimensions + [height, width, channel]`.
    psf_lateral: 1D `tf.Tensor` representing lateral PSF of imaging device.
      Has shape `[psf_length, in_channels, out_channels]`.
    psf_axial: Same as `psf_lateral` but for axial psf.
    lateral_spec: Optional parameters to modify convolution in lateral
      direction. See `complex_convolution` for options.
    axial_spec: Optional parameters to modify convolution in axial
      direction. See `complex_convolution` for options.

  Returns:
    A `tf.Tensor` of the same shape as `state` corresponding to the signal
    generated by the simulated imaging of `state`.

  Raises:
    ValueError: If any argument has the incorrect dimensions.
  """
  if state.shape.ndims < 3:
    raise ValueError(
      "State must be at least 3D (`[width, height, batch]`) but got {}"
      "".format(state.shape))

  # First convolve with `psf_lateral`.
  state = complex_convolution.convolve_complex_1d(state, psf_lateral,
                                                  padding="SAME")

  axes = [axis for axis in range(state.shape.ndims)]
  spatial_axes_permute = axes[:-3] + [axes[-2]] + [axes[-3]] + [axes[-1]]

  # Swap `height` and `width` dimensions. Now has shape
  # `batch + [width, height, channel]`. This effectively makes `width` a
  # batch dimension.
  state = tf.transpose(state, spatial_axes_permute)

  # Convolve with `psf_axial`.
  state = complex_convolution.convolve_complex_1d(state, psf_axial,
                                                  padding="SAME")

  # Return `height` and `width` axes to original locations.
  # State now has shape `batch + [height, width, out_channel]`
  state = tf.transpose(state, spatial_axes_permute)

  # Return real values.
  return tf.abs(state)


def rotate_and_observe(
    state,
    angles,
    psf_lateral_filter,
    psf_axial_filter,
):
  """Convenience function to perform rotation, observe, and reverse rotation.

  Args:
    state: `tf.Tensor` of shape `[batch, height, width, channels]`.
    angles: Angles to rotate `state` by before observation.
      See `tensor_utils.rotate_tensor`
    psf_lateral_filter: See `observe`.
    psf_axial_filter: See `observe`.

  Returns:
    `tf.Tensor` of shape `[batch, angle_count, height, width, channels]`.
    Represents simulated RF signal acquired from observing `state` using given
    psf's at `angles`.
  """
  if state.shape.ndims != 3:
    raise ValueError(
      "`state` must be 3D (`[batch, height, width]`."
      " Got {})".format(state.shape.as_list()))

  state = state[:, tf.newaxis, :, :, tf.newaxis]
  state = tf.tile(state, [1, angles.shape[0], 1, 1, 1])

  # Rotate images along `angle` dimension.
  state = tensor_utils.rotate_tensor(state, angles, 1)

  # Convert to complex dtype.
  state = tf.cast(state, tf.complex64)

  # Observation.
  state = observe(state, psf_lateral_filter, psf_axial_filter)

  # Rotate back.
  state = tensor_utils.rotate_tensor(state, -1 * angles, 1)

  # Return intensity.
  return state