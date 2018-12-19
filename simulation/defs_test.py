"""Tests for `defs.py`"""

import unittest

from simulation import defs

def simple_observation_spec(
    angles=[0, 1., 2.],
    frequencies=[3.e6, 4.e6, 5.e6],
    grid_dimension=.5e-4,
    transducer_bandwidth=1.,
    numerical_aperture=2.,
):
  return defs.ObservationSpec(
    angles, frequencies, grid_dimension, transducer_bandwidth,
    numerical_aperture)


class defsTest(unittest.TestCase):

  def test_simple_observation(self):
    simple_observation_spec()


if __name__ == "__main__":
  unittest.main()