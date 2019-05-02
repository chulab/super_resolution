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
      array=tf.Variable(psf_temp, dtype=np.float32, trainable=)
    )

    psfs.append(psf)

  return psfs


def observation_from_distribution(distribution, psfs, grid_dimension):
  """Computes simulated US image using `psfs`."""
  sampling_rate_time = (grid_dimension / defs._SPEED_OF_SOUND_TISSUE) ** -1
  observations = [
      tf_fft_conv.signal_and_envelope(
        distribution,
        psf.array,
        mode='same',
        sampling_rate=sampling_rate_time,
        frequency=psf.psf_description.frequency,
        angle=psf.angle * np.pi / 180,
        freq_sigma=psf.psf_description.frequency_sigma * 2,
      ) for psf in psfs]
  # Extract envelopes from `(envelope, signal)` tuple.
  envelopes = [o[0] for o in observations]
  return tf.stack(envelopes, -1)[tf.newaxis]
