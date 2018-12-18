"""Common defintions."""

from collections import namedtuple
import numpy as np

_SPEED_OF_SOUND_WATER = 1498  # m/s


class ObservationSpec(namedtuple(
  'ObservationSpec', ['angles', 'frequencies', 'grid_dimension',
                      'transducer_bandwidth', 'numerical_aperture'])):
  """ObservationSpec contains parameters associated with US observation."""

  def __new__(cls, angles, frequencies, grid_dimension, transducer_bandwidth,
              numerical_aperture):
    assert isinstance(angles, list)
    if not all(0. <= angle < np.pi for angle in angles):
      raise ValueError("All angle in `angles` must be scalars between 0 and "
                       "pi. Got {}.".format(angles))

    assert isinstance(frequencies, list)

    assert isinstance(grid_dimension, float)

    assert isinstance(transducer_bandwidth, float)

    assert isinstance(numerical_aperture, float)

    return super(ObservationSpec, cls).__new__(
      cls, angles, frequencies, grid_dimension, transducer_bandwidth,
      numerical_aperture)