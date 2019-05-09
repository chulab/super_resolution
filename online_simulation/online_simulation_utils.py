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
  """Generates set of `PsfDescription`.

  This function generates a set of `PsfDescription` which consists of all
    combinations:
    - `angle_count` angles between 0. and `angle_limit` degrees.
    - `frequency_count` frequencies between `min_frequency` and `max_frequency`.
    - `mode_count` modes starting at 0th order mode (0, ... , `mode_count`).
  """
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
  with tf.name_scope("psf"):
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
      psf_temp = psf_temp.astype(np.float32)
      # Swap `x` and `z` axes.
      psf_temp = np.transpose(psf_temp, [1, 0])

      psf = defs.PSF(
        psf_description=d,
        angle=a,
        array=tf.get_variable(
          name="frequency_{frequency}_angle_{angle}_mode_{mode}".format(
            frequency=d.frequency,
            angle=a,
            mode=d.mode,
          ),
          initializer=psf_temp,
          dtype=tf.float32,
          trainable=False)
      )
      psfs.append(psf)
    return psfs


class USSimulator():

  def __init__(self, psfs, image_grid_size, grid_dimension):
    self.psfs = psfs
    self.image_grid_size = image_grid_size
    self.dft_size = self._dft_size(image_grid_size)
    self.sampling_rate_time = (
      (grid_dimension / defs._SPEED_OF_SOUND_TISSUE) ** -1)

    self.grid = self._ft_grid(self.sampling_rate_time, self.dft_size)

  def _dft_size(self, image_grid_size):
    return [s * 2 - 1 for s in image_grid_size]

  def _fslice(self, image_grid_size, psf_grid_size):
    out_shape = [s1 + s2 - 1 for s1, s2 in zip(image_grid_size, psf_grid_size)]
    return tuple([slice(sz) for sz in out_shape])

  def _ft_grid(self, sampling_rate, dft_size):
    with tf.name_scope("ft_grid"):
      ft_grid_unit = sampling_rate / dft_size[0]
      length_ft = [(s - 1) * ft_grid_unit for s in dft_size]
      grid = response_functions.coordinate_grid(
        length_ft, [ft_grid_unit] * len(length_ft), center=True, mode="NUMPY")
      return tf.Variable(np.stack([g.astype(np.float32) for g in grid], -1),
                         trainable=False)

  def _filter(self, grid, frequency, angle, frequency_sigma):
    with tf.name_scope("filter"):
      angle = angle * np.pi / 180.
      return tf_fft_conv.centered_filter(
          grid, frequency * 2, angle, frequency_sigma * 4) * 2

  def ft_tensor(self, tensor, dft_size):
    with tf.name_scope("ft_tensor"):
      tensor = tf_fft_conv.pad_to_size(tensor, dft_size)
      tensor = tf_fft_conv.convert_to_complex(tensor)
      return tf.fft2d(tensor)

  def signal_and_envelope(self, ft_distribution, ft_psf, filter, fslice, mode):
    """Compute RF signal and envelope for simulation.

    This function computes the signal in the frequency domain by multiplying
    `ft_distribution` and `ft_psf` and `filter`.
    """
    with tf.name_scope("signal_and_envelope"):
      tensor = ft_distribution * ft_psf
      filter = tf_fft_conv.ifftshift_split(filter)
      filter = tf_fft_conv.convert_to_complex(filter)
      tensor = tensor * filter
      tensor = tf.ifft2d(tensor)[fslice]
      envelope_tensor_out = tf.abs(tensor)
      signal_tensor_out = tf.real(tensor)

      if mode == 'full':
        return envelope_tensor_out, signal_tensor_out
      if mode == 'same':
        return (
          tf_fft_conv._centered(envelope_tensor_out, self.image_grid_size),
          tf_fft_conv._centered(signal_tensor_out, self.image_grid_size))

  def observation_from_distribution(self, distribution):
    with tf.name_scope("simulation"):
      ft_distribution = self.ft_tensor(distribution, self.dft_size)
      observations = [
        self.signal_and_envelope(
          ft_distribution=ft_distribution,
          ft_psf=self.ft_tensor(psf.array, self.dft_size),
          filter=self._filter(
              self.grid,
              psf.psf_description.frequency,
              psf.angle,
              psf.psf_description.frequency_sigma * 2
          ),
          fslice=self._fslice(
            distribution.shape.as_list(), psf.array.shape.as_list()),
          mode="same"
        ) for psf in self.psfs]

      with tf.name_scope("extract_envelopes"):
        # Extract envelopes from `(envelope, signal)` tuple.
        envelopes = [o[0] for o in observations]
        return tf.stack(envelopes, -1)
