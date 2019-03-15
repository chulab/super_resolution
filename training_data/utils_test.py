"""Test for `training_data/utils.py`"""

import numpy as np
from training_data import utils
import unittest

from simulation import defs

class utilsTest(unittest.TestCase):

  def test_extract_freq_angles(self):
    descriptions = [
      defs.PsfDescription(1e6, 2, .1e6, .1),
      defs.PsfDescription(2e6, 5, .4e6, .3),
      defs.PsfDescription(3e6, 1, .5e6, .4)
    ]
    angles = [np.random.rand(1) for _ in range(7)]
    observation_spec = defs.ObservationSpec(.1, angles, descriptions)
    tensors = [np.random.rand(24, 24, 3) for _ in range(7)]
    tensor = np.stack(tensors, 0)

    images = utils.extract_angles_and_frequencies(tensor, observation_spec)

    i = 0
    for c_a, angle in enumerate(angles):
      for c_d, description in enumerate(descriptions):
        np.testing.assert_equal(tensor[c_a, :, :, c_d], images[i].image)
        self.assertEqual(angle, images[i].angle)
        self.assertCountEqual(description, images[i].psf_description)
        i += 1

if __name__ == "__main__":
  unittest.main()