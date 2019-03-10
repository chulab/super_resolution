"""Tests for `simulate.py`."""

import unittest

import numpy as np

from simulation import simulate
from simulation import defs


class SimulationTest(unittest.TestCase):

  def _make_simulator(
      self,
      angles=[0, np.pi],
      frequencies=[2e6, 4e6, 6e6],
      modes=[0,1],
      numerical_aperture=.125,
      frequency_sigmas=[2e5]*3,
      psf_axial_length=1e-3,
      psf_transverse_length=3e-3,
      grid_unit=2.5e-5,
  ):
    return simulate.USSimulator(
      angles=angles,
      frequencies=frequencies,
      modes=modes,
      numerical_aperture=numerical_aperture,
      frequency_sigmas=frequency_sigmas,
      psf_axial_length=psf_axial_length,
      psf_transverse_length=psf_transverse_length,
      grid_unit=grid_unit,
    )

  def testInitialize(self):
    self._make_simulator()

  def testGet(self):
    angles = np.random.rand(5)
    frequencies = np.random.rand(4) * 1e6
    modes=[0, 1, 2]
    grid_unit=3.3e-5
    sim_test = self._make_simulator(
      angles=angles,
      frequencies=frequencies,
      modes=modes,
      grid_unit=grid_unit
    )
    np.testing.assert_equal(angles, sim_test.angles)
    np.testing.assert_equal(frequencies, sim_test.frequencies)
    np.testing.assert_equal(modes, sim_test.modes)
    np.testing.assert_equal(grid_unit, sim_test.grid_unit)

  def testCannotSetFrequencies(self):
    sim_test = self._make_simulator()
    with self.assertRaises(AttributeError):
      sim_test.frequencies = 1.2

  def testCannotSetModes(self):
    sim_test = self._make_simulator()
    with self.assertRaises(AttributeError):
      sim_test.modes = 1.3

  def testSetAngles(self):
    old_angles = [1, 2, 3]
    sim_test = self._make_simulator(angles=[1, 2, 3])
    np.testing.assert_equal(old_angles, sim_test.angles)

    new_angles = [5, 6, 7]
    sim_test.angles = new_angles
    np.testing.assert_equal(new_angles, sim_test.angles)

  def testPsfdescription(self):
    sim_test = self._make_simulator()
    frequencies=[1e2, 2e2]
    modes=[1, 3, 5]
    frequency_sigmas=[.7, .3,]
    numerical_apertures=[7., 8.,]
    descriptions = sim_test._generate_psf_description(
      frequencies, modes, frequency_sigmas, numerical_apertures)

    real_descriptions = [
      defs.PsfDescription(1e2, 1, .7, 7.),
      defs.PsfDescription(1e2, 3, .7, 7.),
      defs.PsfDescription(1e2, 5, .7, 7.),
      defs.PsfDescription(2e2, 1, .3, 8.),
      defs.PsfDescription(2e2, 3, .3, 8.),
      defs.PsfDescription(2e2, 5, .3, 8.),
        ]

    [np.testing.assert_equal(truth._asdict(), const._asdict()) for
     truth, const in zip(sorted(real_descriptions), sorted(descriptions))]

  def testSimulation(self):
    sim_test = self._make_simulator(
      angles=[1] * 4,
      frequencies=[2e6] * 3,
      modes=[0] * 2,
    )
    sample_scatterers = np.random.rand(2, 25, 25).astype(np.float32)
    us_images = sim_test.simulate(sample_scatterers)
    np.testing.assert_equal([2, 4, 25, 25, 6], us_images.shape)

if __name__ == "__main__":
  unittest.main()