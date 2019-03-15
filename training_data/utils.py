"""Utils for processing training data."""

import numpy as np

from typing import List

from simulation import defs
from utils import array_utils

def extract_angles_and_frequencies(
    tensor: np.ndarray,
    observation_spec: defs.ObservationSpec,
) -> List[defs.USImage]:
  """Extracts angles and frequencies from observation tensor.

  An Observation tensor contains the result of simulation on a distribution
  using an `USSimulator`. This function breaks up the Observation into a set
  of `USImage` objects which contain the original observation and the angle
  and description.

  Args:
    Tensor with shape `[angles, height, width, channels]`.

  Returns:
    List of tuples containing the rf image, angle, and psf_description.
  """
  # First extract set of images from each angle.
  split_by_angles = array_utils.reduce_split(tensor, 0)
  images = []
  for angle, tensor in zip(observation_spec.angles, split_by_angles):
    # Split by `psf_description`.
    images += [
      defs.USImage(image=image, angle=angle, psf_description=psf_description)
      for image, psf_description in zip(array_utils.reduce_split(tensor, -1),
                   observation_spec.psf_descriptions)]
  return images