"""Tests for `defs.py`"""

import unittest

from simulation import defs

def simple_observation_spec(
    angles=[0, 1., 2.],
    frequencies=[3.e6, 4.e6, 5.e6],
    modes=[0, 1],
    grid_dimension=.5e-4,
    transducer_bandwidth=1.,
    numerical_aperture=2.,
):
  return defs.ObservationSpec(
    angles, frequencies, modes, grid_dimension, transducer_bandwidth,
    numerical_aperture)


class defsTest(unittest.TestCase):

  def test_simple_observation(self):
    simple_observation_spec()

  def testModes(self):
    modes = [2, 3]
    os = simple_observation_spec(modes=modes)
    [self.assertEqual(true_mode, os_mode) for true_mode, os_mode in zip(
      modes, os.modes
    )]

  def testWavelengthFromFrequency(self):
    wavelength = 2.496e-4
    frequency = 6e6
    self.assertAlmostEqual(
      wavelength, defs.wavelength_from_frequency(frequency), places=3)

  def testFrequencyFromWavelength(self):
    wavelength = 7.49e-4
    frequency = 2e6
    self.assertAlmostEqual(
      frequency, defs.frequency_from_wavelength(wavelength), places=3)


if __name__ == "__main__":
  unittest.main()