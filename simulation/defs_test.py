"""Tests for `defs.py`"""

import unittest

import numpy as np

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


class testObservationSpec(unittest.TestCase):

  def test_simple_observation(self):
    simple_observation_spec()

  def testModes(self):
    modes = [2, 3]
    os = simple_observation_spec(modes=modes)
    [self.assertEqual(true_mode, os_mode) for true_mode, os_mode in zip(
      modes, os.modes
    )]

  def testBadModeNotInteger(self):
    modes = [3, 1.3]
    with self.assertRaisesRegex(ValueError, "Modes must be integers"):
      simple_observation_spec(modes=modes)

  def testBadModeLessthanZero(self):
    modes = [0, 1, -2]
    with self.assertRaisesRegex(ValueError, "Modes must be integers"):
      simple_observation_spec(modes=modes)


class testWavelengthFrequency(unittest.TestCase):

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


class testPSFDescription(unittest.TestCase):

  def psf_description(self, frequency, mode, frequency_sigma,
                      numerical_aperture):
    return defs.PsfDescription(frequency, mode, frequency_sigma,
                               numerical_aperture)

  def testPsfDescription(self):
    frequency = 1.7e6
    mode=2
    frequency_sigma=.1e6
    numerical_aperture=.1
    psf_description = self.psf_description(
      frequency, mode, frequency_sigma, numerical_aperture)
    self.assertEqual(psf_description.frequency, frequency)
    self.assertEqual(psf_description.mode, mode)
    self.assertEqual(psf_description.frequency_sigma, frequency_sigma)
    self.assertEqual(psf_description.numerical_aperture, numerical_aperture)

  def testBadFrequency(self):
    frequency = -.5e3
    mode=1
    frequency_sigma=.1e6
    numerical_aperture=.1
    with self.assertRaises(AssertionError):
      self.psf_description(
        frequency, mode, frequency_sigma, numerical_aperture)

  def testBadMode(self):
    frequency = .5e3
    mode = .2
    frequency_sigma=.1e6
    numerical_aperture=.1
    with self.assertRaises(AssertionError):
      self.psf_description(
        frequency, mode, frequency_sigma, numerical_aperture)

  def testbadSigmaFrequency(self):
    frequency = .5e3
    mode = 2
    frequency_sigma=-.1e6
    numerical_aperture=.1
    with self.assertRaises(AssertionError):
      self.psf_description(
        frequency, mode, frequency_sigma, numerical_aperture)

  def testbadNA(self):
    frequency = .5e3
    mode = 2
    frequency_sigma=.1e6
    numerical_aperture=-.1
    with self.assertRaises(AssertionError):
      self.psf_description(
        frequency, mode, frequency_sigma, numerical_aperture)


class testPSF(unittest.TestCase):

  def _PSF(self, psf_description, physical_size, array):
    return defs.PSF(psf_description, physical_size, array)

  def testPSF(self):
    psf_description = defs.PsfDescription(1e6, 2, .1e6, .1)
    physical_size = (1e-3, 2e-3)
    array = np.random.rand(14, 25)
    psf = self._PSF(psf_description, physical_size, array)

    np.testing.assert_equal(
      psf_description._asdict(), psf.psf_description._asdict())

    self.assertEqual(physical_size, psf.physical_size)
    np.testing.assert_equal(array, psf.array)

  def testPSFBadpsfDescription(self):
    psf_description = [1e6, 2, .1e6, .1]
    physical_size = (1e-3, 2e-3)
    array = np.random.rand(14, 25)
    with self.assertRaises(AssertionError):
      self._PSF(psf_description, physical_size, array)

  def testPSFBadArrayPhysicalSize(self):
    psf_description = defs.PsfDescription(1e6, 2, .1e6, .1)
    physical_size = (-1e-3, 2e-3)
    array = np.random.rand(14, 25)
    with self.assertRaises(AssertionError):
      self._PSF(psf_description, physical_size, array)

  def testPSFBadArrayPhsicalSizeDim(self):
    psf_description = defs.PsfDescription(1e6, 2, .1e6, .1)
    physical_size = (1e-3, 2e-3)
    array = np.random.rand(14, 25, 12)
    with self.assertRaises(AssertionError):
      self._PSF(psf_description, physical_size, array)

  def testPSFBadArray(self):
    psf_description = defs.PsfDescription(1e6, 2, .1e6, .1)
    physical_size = (1e-3, 2e-3)
    array = np.random.rand(14, 25, 12) # Too many dimensions
    with self.assertRaises(AssertionError):
      self._PSF(psf_description, physical_size, array)


if __name__ == "__main__":
  unittest.main()