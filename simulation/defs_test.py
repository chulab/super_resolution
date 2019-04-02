"""Tests for `defs.py`"""

import unittest

import numpy as np

from simulation import defs


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


class testObservationSpec(unittest.TestCase):

  def simple_observation_spec(self, grid_dimension, angles, psf_descriptions,
  ):
    return defs.ObservationSpec(
      grid_dimension=grid_dimension,
      angles=angles,
      psf_descriptions=psf_descriptions,
    )

  def test_simple_observation(self):
    grid_dimension = .3
    print(isinstance(grid_dimension, float))
    angles = [a for a in np.random.rand(5)]
    psf_descriptions = [
      defs.PsfDescription(1e6, 2, .1e6, .1),
      defs.PsfDescription(1e6, 2, .1e6, .1),
    ]
    observation_spec = self.simple_observation_spec(
      grid_dimension, angles, psf_descriptions
    )

    self.assertEqual(grid_dimension, observation_spec.grid_dimension)
    self.assertCountEqual(angles, observation_spec.angles)
    self.assertCountEqual(psf_descriptions, observation_spec.psf_descriptions)

  def testBadGridDimension(self):
    grid_dimension = -.3
    angles = [a for a in np.random.rand(5)]
    psf_descriptions = [
      defs.PsfDescription(1e6, 2, .1e6, .1),
      defs.PsfDescription(1e6, 2, .1e6, .1),
    ]
    with self.assertRaisesRegex(ValueError, "`grid_dimension` must be"):
      self.simple_observation_spec(grid_dimension, angles, psf_descriptions)

  def testBadAngles(self):
    grid_dimension = .3
    angles = [3.5]
    psf_descriptions = [
      defs.PsfDescription(1e6, 2, .1e6, .1),
      defs.PsfDescription(1e6, 2, .1e6, .1),
    ]
    with self.assertRaisesRegex(ValueError, "All angle in `angles`"):
      self.simple_observation_spec(grid_dimension, angles, psf_descriptions)

  def testBadPsfDescription(self):
    grid_dimension = .3
    angles = [1., 2.]
    psf_descriptions = [
      defs.PsfDescription(1e6, 2, .1e6, .1),
      "NOTADESCRIPTION"
    ]
    with self.assertRaisesRegex(
        ValueError, "All elements in `psf_descriptions`"):
      self.simple_observation_spec(grid_dimension, angles, psf_descriptions)

class testUSImage(unittest.TestCase):

  def simple_US_image(self, image, angle, psf_description,
  ):
    return defs.USImage(
      image=image,
      angle=angle,
      psf_description=psf_description,
    )

  def test_simple_usimage(self):
    image = np.random.rand(20, 40)
    angle = 1.4
    psf_description = defs.PsfDescription(1e6, 2, .1e6, .1)

    usimage = self.simple_US_image(
      image, angle, psf_description
    )

    np.testing.assert_equal(image, usimage.image)
    self.assertEqual(angle, usimage.angle)
    self.assertCountEqual(psf_description, usimage.psf_description)


if __name__ == "__main__":
  unittest.main()