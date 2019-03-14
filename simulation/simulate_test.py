"""Tests for `simulate.py`."""

import unittest

import numpy as np

from simulation import simulate
from simulation import defs


class SimulationTest(unittest.TestCase):

  def _make_simulator(
      self,
      grid_unit,
      angles,
      psf_descriptions,
      psf_axial_length,
      psf_transverse_length,
  ):
    return simulate.USSimulator(
      grid_unit=grid_unit,
      angles=angles,
      psf_descriptions=psf_descriptions,
      psf_axial_length=psf_axial_length,
      psf_transverse_length=psf_transverse_length,
    )

  def testGet(self):
    angles = np.random.rand(5)
    frequencies = np.random.rand(4) * 1e6
    frequency_sigmas = np.random.rand(4) * 1e5
    numerical_apertures = np.random.rand(4)
    modes=[0, 1, 2, 3]
    grid_unit=3.3e-5

    descriptions = [
      defs.PsfDescription(f, m, fs, na) for f, m, fs, na in zip(
        frequencies, modes, frequency_sigmas, numerical_apertures
      )
    ]

    sim_test = self._make_simulator(
      angles=angles,
      grid_unit=grid_unit,
      psf_descriptions=descriptions,
      psf_transverse_length=2.e-4,
      psf_axial_length=1.e-4,
    )
    np.testing.assert_equal(angles, sim_test.angles)
    np.testing.assert_equal(grid_unit, sim_test.grid_unit)
    [self.assertSequenceEqual(true, loaded) for true, loaded in zip(
      descriptions, sim_test.psf_descriptions)
     ]

  def testRotateBack(self):
    angles = np.random.rand(5)
    frequencies = np.random.rand(4) * 1e6
    frequency_sigmas = np.random.rand(4) * 1e5
    numerical_apertures = np.random.rand(4)
    modes=[0, 1, 2, 3]
    grid_unit=3.3e-5

    descriptions = [
      defs.PsfDescription(f, m, fs, na) for f, m, fs, na in zip(
        frequencies, modes, frequency_sigmas, numerical_apertures
      )
    ]

    sim_test = self._make_simulator(
      angles=angles,
      grid_unit=grid_unit,
      psf_descriptions=descriptions,
      psf_transverse_length=2.e-4,
      psf_axial_length=1.e-4,
    )
    np.testing.assert_equal(False, sim_test.rotate_back)

    sim_test.rotate_back = True

    np.testing.assert_equal(True, sim_test.rotate_back)

  def testSetAngles(self):
    angles = np.random.rand(5)
    frequencies = np.random.rand(4) * 1e6
    frequency_sigmas = np.random.rand(4) * 1e5
    numerical_apertures = np.random.rand(4)
    modes=[0, 1, 2, 3]
    grid_unit=3.3e-5

    descriptions = [
      defs.PsfDescription(f, m, fs, na) for f, m, fs, na in zip(
        frequencies, modes, frequency_sigmas, numerical_apertures
      )
    ]

    sim_test = self._make_simulator(
      angles=angles,
      grid_unit=grid_unit,
      psf_descriptions=descriptions,
      psf_transverse_length=2.e-4,
      psf_axial_length=1.e-4,
    )
    np.testing.assert_equal(angles, sim_test.angles)

    new_angles = [5, 6, 7]
    sim_test.angles = new_angles
    np.testing.assert_equal(new_angles, sim_test.angles)

  def testSimulation(self):
    angles = np.random.rand(5)
    frequencies = np.random.rand(4) * 1e6
    frequency_sigmas = np.random.rand(4) * 1e5
    numerical_apertures = np.random.rand(4)
    modes=[0, 1, 2, 3]
    grid_unit=3.3e-5

    descriptions = [
      defs.PsfDescription(f, m, fs, na) for f, m, fs, na in zip(
        frequencies, modes, frequency_sigmas, numerical_apertures
      )
    ]

    sim_test = self._make_simulator(
      angles=angles,
      grid_unit=grid_unit,
      psf_descriptions=descriptions,
      psf_transverse_length=2.e-4,
      psf_axial_length=1.e-4,
    )
    np.testing.assert_equal(angles, sim_test.angles)

    sample_scatterers = np.random.rand(2, 25, 25).astype(np.float32)
    us_images = sim_test.simulate(sample_scatterers)
    np.testing.assert_equal([2, 5, 25, 25, 4], us_images.shape)

if __name__ == "__main__":
  unittest.main()