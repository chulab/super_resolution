"""Tests for `record_writer.py`"""

import glob
from collections import namedtuple
import shutil
import tempfile
import numpy as np


import tensorflow as tf

from training_data import record_writer
from training_data import record_utils
from simulation import defs

class TestRecordWriter(tf.test.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test
    shutil.rmtree(self.test_dir)

  def make_record_writer(self, observation_spec, directory, dataset_name,
                         examples_per_shard):
    return record_writer.RecordWriter(
      directory, dataset_name, examples_per_shard)

  def testRecordWriter(self):
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(self.test_dir, name,
                                    examples_per_shard)
    self.assertEqual(name, rw.dataset_name)
    self.assertEqual(examples_per_shard, rw.examples_per_shard)
    self.assertEqual(self.test_dir, rw.directory)

  def testSetDirectory(self):
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(self.test_dir, name,
                                    examples_per_shard)
    new_directory = tempfile.mkdtemp()
    rw.directory = new_directory
    shutil.rmtree(new_directory)
    self.assertEqual(new_directory, rw.directory)

  def testSetInvalidDirectory(self):
    name = "test_name"
    examples_per_shard = 23
    invalid_directory = "blahblahbla"
    with self.assertRaisesRegex(ValueError, "Directory is not valid"):
      record_writer.RecordWriter(invalid_directory, name,
                                    examples_per_shard)

  def testSetInvalidExamplesPerShard(self):
    name = "test_name"
    examples_per_shard = -2
    with self.assertRaisesRegex(ValueError, "`examples_per_shard` must be "
                                            "positive integer"):
      record_writer.RecordWriter(self.test_dir, name,
                                    examples_per_shard)

  def testSave(self):
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(self.test_dir, name,
                                    examples_per_shard)

    test_distribution = np.random.rand(20, 25).astype(np.float32)
    test_observation = np.random.rand(30, 50).astype(np.float32)
    rw.save(test_distribution, test_observation)
    self.assertEqual(1, rw._current_example_in_shard)
    rw.close()

    files = glob.glob(self.test_dir + "/*")
    # Check written file has expected name.
    self.assertEqual([self.test_dir+"/"+name+"_0"], files)

    test_dataset = tf.data.TFRecordDataset(files)
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    next_dist, next_obs = iterator.get_next()

    with self.test_session() as sess:
      dist_eval, obs_eval = sess.run(
        [next_dist, next_obs])

      self.assertAllEqual(dist_eval.shape, (20, 25))
      self.assertAllEqual(obs_eval.shape, (30, 50))

      self.assertAllEqual(test_distribution, dist_eval)
      self.assertAllEqual(test_observation, obs_eval)

  def testWriteCorrectNumber(self):
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(self.test_dir, name, examples_per_shard)

    distributions = []
    observations = []
    for i in range(54):
      test_distribution = np.random.rand(20, 25).astype(np.float32)
      test_observation = np.random.rand(30, 50).astype(np.float32)
      rw.save(test_distribution, test_observation)
      distributions.append(test_distribution)
      observations.append(test_observation)
    self.assertEqual(8, rw._current_example_in_shard)
    self.assertEqual(2, rw._current_shard)
    rw.close()

    files = glob.glob(self.test_dir + "/*")
    # Check written file has expected name.
    self.assertEqual(sorted([self.test_dir+"/"+name+"_0", self.test_dir+"/"+name+"_1",
                      self.test_dir + "/" + name + "_2"]), sorted(files))

    test_dataset = tf.data.TFRecordDataset(sorted(files))
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    next_dist, next_obs = iterator.get_next()

    with self.test_session() as sess:
      for i in range(52):
        dist_eval, obs_eval = sess.run(
          [next_dist, next_obs])

        self.assertAllEqual(distributions[i], dist_eval)
        self.assertAllEqual(observations[i], obs_eval)

  def testWriterClosed(self):
    pass


if __name__ == "__main__":
  tf.test.main()