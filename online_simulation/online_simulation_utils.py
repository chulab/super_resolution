"""Utilities for performing online simulation."""
from typing import List

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
    psf_descriptions=None, # (angle, frequency, mode) tuples
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
    psf_dimension:float,
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
      array=psf_temp.astype(np.float32)
    )

    psfs.append(psf)

  return psfs


class USsimulator():

  def __init__(self, psfs):
    self.psfs = psfs

  def simulate(self, distribution):
    return [tf_fft_conv.fft_correlate(distribution, psf, mode='same') for psf in self.psfs]

def observation_from_distribution(sim, distribution):
  observations = sim.simulate(distribution)
  return tf.stack(observations, -1)[tf.newaxis]