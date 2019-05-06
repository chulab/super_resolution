"""Utilities for performing online simulation."""
import itertools

from scipy.spatial import transform

import tensorflow as tf
import numpy as np

from simulation import response_functions
from simulation import defs
from utils import tf_fft_conv


def simulation_params():
  return tf.contrib.training.HParams(
    psf_dimension=2e-3,
    psf_descriptions=None,  # (angle, frequency, mode) tuples
    frequency_sigma=1e6,
    numerical_aperture=.125,
  )


def grid_psf_descriptions(
    angle_limit,
    angle_count,
    min_frequency,
    max_frequency,
    frequency_count,
    mode_count,
    numerical_aperture,
    frequency_sigma,
):
  angles = np.linspace(0., angle_limit, angle_count)
  frequencies = np.linspace(min_frequency, max_frequency, frequency_count)
  modes = list(range(mode_count))

  descriptions = []

  for f, m in itertools.product(frequencies, modes):
    descriptions.append(
      defs.PsfDescription(
        frequency=f,
        mode=m,
        frequency_sigma=frequency_sigma,
        numerical_aperture=numerical_aperture,
      )
    )

  return list(itertools.product(angles, descriptions))


def _coordinates(lengths, grid_unit, rotation_angle):
  grid_dimensions = [grid_unit, grid_unit, grid_unit]

  # Define coordinates of rotated frame.
  coordinates = np.stack(
    response_functions.coordinate_grid(lengths, grid_dimensions, center=True),
    -1)

  # Rotate coordinates
  rotation = transform.Rotation.from_euler('y', [rotation_angle], degrees=True)
  original_coordinates_shape = coordinates.shape
  coordinates_temp = np.reshape(coordinates, [-1, 3])
  coordinates_rot = rotation.apply(coordinates_temp)
  coordinates_rot = np.reshape(coordinates_rot, original_coordinates_shape)

  coordinates_scale = coordinates_rot * np.array([1, 1, 2.])

  return coordinates_scale


def make_psf(
    psf_dimension: float,
    grid_dimension: float,
    descriptions,
):
  psfs = []

  for a, d in descriptions:
    lengths = [psf_dimension, 0, psf_dimension]

    coordinates = _coordinates(lengths, grid_dimension, a)

    psf_temp = response_functions.gaussian_impulse_response_v2(
      coordinates=coordinates,
      frequency=d.frequency,
      mode=d.mode,
      numerical_aperture=d.numerical_aperture,
      frequency_sigma=d.frequency_sigma,
    )[:, 0, :]

    # Swap `x` and `z` axes.
    psf_temp = np.transpose(psf_temp, [1, 0])

    psf = defs.PSF(
      psf_description=d,
      angle=a,
      array=tf.Variable(psf_temp, dtype=np.float32, trainable=False)
    )

    psfs.append(psf)

  return psfs


def signal_and_envelope(tensor_a, tensor_b, pulse_filter, mode):
  """Calculates analytic signal of convolving `tensor_a` and `tensor_b`.

  Size of `tensor_b` must be less than or equal to `tensor_a`.

  This function is used to calculate the envelope containing speckle resulting
  from convolving a scatterer distribution (`tensor_a`) with a psf (`tensor_b`).

  args:
    tensor_a: `tf.Tensor` of shape `[H, W]`
    tensor_b: `tf.Tensor` of shape `[H2, W2]`
    mode: See documentation for `fft_conv`.
    sampling_rate: Sampling rate in Hz of `tensor_a`.
    frequency: Center frequency to filter.
    angle: Incident angle of impulse in radians.
    frequency_sigma: Approximate sigma in frequency of pulse.

  Returns: `tf.Tensor`
  """
  tensor_size_a = tensor_a.shape.as_list()
  tensor_size_b = tensor_b.shape.as_list()
  dft_size = [size * 2 - 1 for size in tensor_size_a]
  out_shape = [s1 + s2 - 1 for s1, s2 in zip(tensor_size_a, tensor_size_b)]
  fslice = tuple([slice(sz) for sz in out_shape])
  tensor_a, tensor_b = tf_fft_conv.pad_to_size(tensor_a, dft_size), tf_fft_conv.pad_to_size(tensor_b,
                                                                    dft_size)
  tensor_a, tensor_b = tf_fft_conv.convert_to_complex(tensor_a), tf_fft_conv.convert_to_complex(
    tensor_b)

  ft_tensor = tf.fft2d(tensor_a) * tf.fft2d(tensor_b)
  fftshift_tensor = tf_fft_conv.fftshift(ft_tensor)
  pulse_filter = tf_fft_conv.convert_to_complex(pulse_filter)
  filtered_tensor = fftshift_tensor * pulse_filter

  filtered_tensor = tf_fft_conv.ifftshift(filtered_tensor)
  envelope_tensor_out = tf.abs(tf.ifft2d(filtered_tensor))[fslice]
  signal_tensor_out = tf.real(tf.ifft2d(ft_tensor))[fslice]

  if mode == 'full':
    return envelope_tensor_out, signal_tensor_out
  if mode == 'same':
    return (tf_fft_conv._centered(envelope_tensor_out, tensor_size_a),
            tf_fft_conv._centered(signal_tensor_out, tensor_size_a))


class USSimulator():

  def __init__(self, psfs, image_grid_size, grid_dimension):
    self.psfs = psfs
    self.dft_size = self._dft_size(image_grid_size)
    self.sampling_rate_time = (grid_dimension / defs._SPEED_OF_SOUND_TISSUE) ** -1

    self.grid = self._ft_grid(self.sampling_rate_time, self.dft_size)

  def _dft_size(self, image_grid_size):
    return [s * 2 - 1 for s in image_grid_size]

  def _ft_grid(self, sampling_rate, dft_size):
    ft_grid_unit = sampling_rate / dft_size[0]
    length_ft = [(s - 1) * ft_grid_unit for s in dft_size]
    grid = response_functions.coordinate_grid(
      length_ft, [ft_grid_unit] * len(length_ft), center=True, mode="NUMPY")
    return tf.Variable(np.stack([g.astype(np.float32) for g in grid], -1), trainable=False)

  def _filter(self, grid, frequency, angle, frequency_sigma):
    return tf_fft_conv.centered_filter(
        grid, frequency * 2, angle, frequency_sigma * 4) * 2

  def observation_from_distribution(self, distribution):
    observations = [
      signal_and_envelope(
        distribution,
        psf.array,
        pulse_filter=self._filter(
            self.grid,
            psf.psf_description.frequency,
            psf.angle,
            psf.psf_description.frequency_sigma * 2
        ),
        mode="same"
      ) for psf in self.psfs]

    # Extract envelopes from `(envelope, signal)` tuple.
    envelopes = [o[0] for o in observations]
    return tf.stack(envelopes, -1)[tf.newaxis]
