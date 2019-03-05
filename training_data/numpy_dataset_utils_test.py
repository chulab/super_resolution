"""Tests for `numpy_dataset_utils.py`"""

import glob
import shutil
import tempfile
import unittest

import numpy as np

from training_data import numpy_dataset_utils


class DatsetUtilsTest(unittest.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test
    shutil.rmtree(self.test_dir)

  def testGeneratorSave(self):
    data = [np.random.rand(40, 40) for _ in range(200)]
    gen = iter(data)
    numpy_dataset_utils.save_dataset(gen, 200, 40, self.test_dir, "test_prefix")
    files = sorted(glob.glob(self.test_dir + "/*"))
    self.assertEqual(len(files), 5)
    loaded = np.concatenate([np.load(file) for file in files], 0)
    np.testing.assert_equal(np.stack(data, 0), loaded)


if __name__ == "__main__":
  unittest.main()