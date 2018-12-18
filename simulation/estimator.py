"""Builds a tf.Estimator model for running US observation simulation."""

import numpy as np
import tensorflow as tf

from simulation import defs
from simulation import psf_utils
from simulation import observation


class SimulationEstimator(tf.estimator.Estimator):
  """Estimator class for running US simulation.

  This class builds a `tf.Estimator` which can be used in `prediction` mode to
  simulate US observation of scatterer distributions fed by dataset `features`.
  """

  def __init__(
      self,
      observation_spec: defs.ObservationSpec,
      axial_psf_length: int,
      lateral_psf_length: int,
      config = None,
  ):
    """Initializes `SimulationEstimator` instance.

    Args:
      observation_spec: `ObservationSpec` describing simulation parameters.
      axial_psf_length: Length of psf used for simulation.
      lateral_psf_length: Same as `axial_psf_length` but for lateral psf.
      config: See `tf.estimator.Estimator`.
    """
    _SPEED_OF_SOUND = defs._SPEED_OF_SOUND_WATER  # m/s
    wavelengths = [_SPEED_OF_SOUND / freq for freq in
                   observation_spec.frequencies]

    # Build lateral filter.
    psf_lateral_filter = psf_utils.lateral_psf_filters(
      lateral_psf_length, wavelengths, observation_spec.numerical_aperture,
      observation_spec.grid_dimension)

    # Build axial filter.
    psf_axial_filter = psf_utils.axial_psf_filters(
    axial_psf_length, wavelengths, observation_spec.numerical_aperture,
      observation_spec.transducer_bandwidth, observation_spec.grid_dimension)

    super(SimulationEstimator, self).__init__(
      model_fn=simulation_model_fn,
      params={
            "psf_lateral_filter": psf_lateral_filter,
            "psf_axial_filter": psf_axial_filter,
            "angles": observation_spec.angles
        },
      config=config)


def simulation_model_fn(features, labels, mode, params):
  """Builds model for ultrasound simulation."""
  if mode is not tf.estimator.ModeKeys.PREDICT:
    raise ValueError('Only `PREDICT` mode is supported, got %s' % mode)
  del labels
  distribution = features
  distribution.shape.assert_is_compatible_with([None, None, None])

  # Convert angles to tensor.
  angles = tf.convert_to_tensor(params["angles"])

  psf_lateral_filter = tf.convert_to_tensor(params["psf_lateral_filter"],
                                            dtype=tf.complex64)

  psf_axial_filter = tf.convert_to_tensor(params["psf_axial_filter"],
                                          dtype=tf.complex64)

  # Simulate observation.
  simulated_observation = observation.rotate_and_observe(
    distribution, angles, psf_lateral_filter, psf_axial_filter)

  # Dictionary of returns.
  predictions = {
      "angles": angles,
      "psf_lateral": psf_lateral_filter,
      "psf_axial": psf_axial_filter,
      "input": distribution,
      "observation": simulated_observation,
  }

  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)