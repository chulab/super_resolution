"""Tests for `create_observation_spec`."""
import json
import unittest
from unittest.mock import patch
import os
import shutil
import sys
import tempfile

from simulation import defs
from simulation import create_observation_spec


class CreateObservationSpecTest(unittest.TestCase):

  def setUp(self):
    # Create a test directory.
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test
    shutil.rmtree(self.test_dir)


  def testCreateObservationSpec(self):
    """Tests saving ObservationSpec using module functions."""
    grid_dimension = .5e-4
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    frequency_sigma = [1.e6, .2e6, 3e6]
    numerical_aperture = [2., 3., 5.]
    modes = [1, 2]

    true_descriptions =[
      defs.PsfDescription(
        frequencies[0], modes[0], frequency_sigma[0], numerical_aperture[0]),
      defs.PsfDescription(
        frequencies[0], modes[1], frequency_sigma[0], numerical_aperture[0]),
      defs.PsfDescription(
        frequencies[1], modes[0], frequency_sigma[1], numerical_aperture[1]),
      defs.PsfDescription(
        frequencies[1], modes[1], frequency_sigma[1], numerical_aperture[1]),
      defs.PsfDescription(
        frequencies[2], modes[0], frequency_sigma[2], numerical_aperture[2]),
      defs.PsfDescription(
        frequencies[2], modes[1], frequency_sigma[2], numerical_aperture[2]),
    ]

    true_spec = defs.ObservationSpec(
      grid_dimension, angles, true_descriptions)

    build_spec = create_observation_spec.observation_spec_from_frequencies_and_modes(
      grid_dimension,
      angles,
      frequencies,
      frequency_sigma,
      numerical_aperture,
      modes,
    )

    self.assertSequenceEqual(true_spec, build_spec)

  def testCreateObservationSpecBadArgs(self):
    grid_dimension = .5e-4
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    frequency_sigma = [1.e6, .2e6, 3e6]
    numerical_aperture = [2., 3.]
    modes = [1, 2]

    with self.assertRaisesRegex(ValueError, "`frequencies`, `frequency_sigma`,"
                                            " and"):
      create_observation_spec.observation_spec_from_frequencies_and_modes(
        grid_dimension, angles, frequencies, frequency_sigma,
        numerical_aperture, modes)

  def testSaveObservationSpec(self):
    """Tests saving ObservationSpec using module functions."""
    grid_dimension = .5e-4
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    frequency_sigma = [1.e6, .2e6, 3e6]
    numerical_aperture = [2., 3., 5.]
    modes = [1, 2]

    build_spec = create_observation_spec.observation_spec_from_frequencies_and_modes(
      grid_dimension,
      angles,
      frequencies,
      frequency_sigma,
      numerical_aperture,
      modes,
    )
    create_observation_spec.save_observation_spec(
      build_spec, self.test_dir, 'test_name'
    )
    file = os.path.join(self.test_dir, 'test_name.json')
    loaded_spec = create_observation_spec.load_observation_spec(file)
    self.assertSequenceEqual(build_spec, loaded_spec)

  def testCreateObservationSpecCLI(self):
    """Tests saving ObservationSpec using CLI."""
    grid_dimension = .5e-4
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    frequency_sigma = [1.e6, .2e6, 3e6]
    numerical_aperture = [2., 3., 5.]
    modes = [1, 2]

    true_spec = create_observation_spec.observation_spec_from_frequencies_and_modes(
      grid_dimension,
      angles,
      frequencies,
      frequency_sigma,
      numerical_aperture,
      modes,
    )

    test_args = ["test_arg",
      "-sd", self.test_dir,
      "-n", 'test_name',
      "-gd", "{}".format(grid_dimension),
      "-a", "{}".format(','.join(str(a) for a in angles)),
      "-f", "{}".format(','.join(str(f) for f in frequencies)),
      "-fs", "{}".format(','.join(str(f) for f in frequency_sigma)),
      "-na", "{}".format(','.join(str(f) for f in numerical_aperture)),
      "-m", "{}".format(','.join(str(m) for m in modes)),
    ]
    with patch.object(sys, 'argv', test_args):
      create_observation_spec.main()
    file = os.path.join(self.test_dir, 'test_name.json')
    loaded_spec = create_observation_spec.load_observation_spec(file)
    self.assertSequenceEqual(true_spec, loaded_spec)

  def testLoadGoogleCloud(self):
    create_observation_spec.load_observation_spec(
      'gs://chu_super_resolution_data/simulation/circle_3_18/observation_spec.json',
    True,
    )


if __name__ == "__main__":
  unittest.main()