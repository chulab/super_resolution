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
      observation_spec, directory, dataset_name, examples_per_shard)

  def testRecordWriter(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(observation_spec, self.test_dir, name,
                                    examples_per_shard)
    self.assertAllEqual(
      observation_spec._asdict(), rw.observation_spec._asdict())
    self.assertEqual(name, rw.dataset_name)
    self.assertEqual(examples_per_shard, rw.examples_per_shard)
    self.assertEqual(self.test_dir, rw.directory)

  def testSetDirectory(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(observation_spec, self.test_dir, name,
                                    examples_per_shard)
    new_directory = tempfile.mkdtemp()
    rw.directory = new_directory
    shutil.rmtree(new_directory)
    self.assertEqual(new_directory, rw.directory)

  def testSetInvalidDirectory(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = 23
    invalid_directory = "blahblahbla"
    with self.assertRaisesRegex(ValueError, "Directory is not valid"):
      record_writer.RecordWriter(observation_spec, invalid_directory, name,
                                    examples_per_shard)

  def testSetInvalidObservationSpec(self):
    fake_os = namedtuple("FakeOS", "foo,bar")
    invalid_observation_spec = fake_os(1, 2)
    name = "test_name"
    examples_per_shard = 23
    with self.assertRaisesRegex(ValueError, "Not a valid `ObservationSpec`"):
      record_writer.RecordWriter(invalid_observation_spec, self.test_dir, name,
                                    examples_per_shard)

  def testSetInvalidExamplesPerShard(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = -2
    with self.assertRaisesRegex(ValueError, "`examples_per_shard` must be "
                                            "positive integer"):
      record_writer.RecordWriter(observation_spec, self.test_dir, name,
                                    examples_per_shard)

  def testSave(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(observation_spec, self.test_dir, name,
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
    next_dist, next_obs, next_obs_params = iterator.get_next()

    with self.test_session() as sess:
      dist_eval, obs_eval, params_eval = sess.run(
        [next_dist, next_obs, next_obs_params])

      self.assertAllEqual(dist_eval.shape, (20, 25))
      self.assertAllEqual(obs_eval.shape, (30, 50))

      self.assertAllEqual(test_distribution, dist_eval)
      self.assertAllEqual(test_observation, obs_eval)
      self.assertAllClose(observation_spec._asdict(), params_eval)

  def testWriteCorrectNumber(self):
    observation_spec = defs.ObservationSpec(
      [0, np.pi / 2], [1.5], [0, 1], .23, .2, .8)
    name = "test_name"
    examples_per_shard = 23
    rw = record_writer.RecordWriter(observation_spec, self.test_dir, name,
                                    examples_per_shard)

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
    next_dist, next_obs, next_obs_params = iterator.get_next()

    with self.test_session() as sess:
      for i in range(52):
        dist_eval, obs_eval, params_eval = sess.run(
          [next_dist, next_obs, next_obs_params])

        self.assertAllEqual(distributions[i], dist_eval)
        self.assertAllEqual(observations[i], obs_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)

  def testWriterClosed(self):
    pass


if __name__ == "__main__":
  tf.test.main()