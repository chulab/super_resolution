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
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    modes = [1, 2]
    grid_dimension = .5e-4
    transducer_bandwidth = 1.
    numerical_aperture = 2.
    true_spec = defs.ObservationSpec(
      angles, frequencies, modes, grid_dimension, transducer_bandwidth,
      numerical_aperture)

    create_observation_spec.save_observation_spec(
      true_spec, self.test_dir, 'test_name'
    )
    file = os.path.join(self.test_dir, 'test_name.json')
    with open(file, 'r') as f:
      loaded_spec = defs.ObservationSpec(**json.load(f))
    self.assertSequenceEqual(loaded_spec, true_spec)


  def testCreateObservationSpecCLI(self):
    """Tests saving ObservationSpec using CLI."""
    angles = [0, 1., 2.]
    frequencies=[3.e6, 4.e2, 5.e6]
    modes = [1, 2]
    grid_dimension=.5e-4
    transducer_bandwidth=1.
    numerical_aperture=2.
    true_spec = defs.ObservationSpec(
      angles, frequencies, modes, grid_dimension, transducer_bandwidth,
      numerical_aperture)

    test_args = ["test_arg",
      "-sd", self.test_dir,
      "-n", 'test_name',
      "-a", "{}".format(','.join(str(a) for a in angles)),
      "-f", "{}".format(','.join(str(f) for f in frequencies)),
      "-m", "{}".format(','.join(str(m) for m in modes)),
      "-gd", "{}".format(grid_dimension),
      "-tb", "{}".format(transducer_bandwidth),
      "-na", "{}".format(numerical_aperture),
    ]
    with patch.object(sys, 'argv', test_args):
      create_observation_spec.main()
    file = os.path.join(self.test_dir, 'test_name.json')
    with open(file) as f:
      loaded_spec = defs.ObservationSpec(**json.load(f))
    self.assertSequenceEqual(loaded_spec, true_spec)


  def testLoadObservationSpec(self):
    """Tests saving ObservationSpec using module functions."""
    angles = [0, 1., 2.]
    frequencies = [3.e6, 4.e2, 5.e6]
    modes = [1, 2]
    grid_dimension = .5e-4
    transducer_bandwidth = 1.
    numerical_aperture = 2.
    true_spec = defs.ObservationSpec(
      angles, frequencies, modes, grid_dimension, transducer_bandwidth,
      numerical_aperture)

    create_observation_spec.save_observation_spec(
      true_spec, self.test_dir, 'test_name'
    )
    file = os.path.join(self.test_dir, 'test_name.json')
    loaded_os = create_observation_spec.load_observation_spec(file)
    self.assertSequenceEqual(true_spec, loaded_os)


if __name__ == "__main__":
  unittest.main()