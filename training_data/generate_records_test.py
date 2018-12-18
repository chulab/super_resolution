"""Tests for `generate_records.py`"""
import glob
import shutil
import tempfile
from parameterized import parameterized

import numpy as np
import tensorflow as tf

from simulation import defs
from training_data import generate_records
from training_data import record_utils

class GenerateRecordsTest(tf.test.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    # Remove the directory after the test
    shutil.rmtree(self.test_dir)

  def convert_dict_to_numpy(self, d):
    for key, value in d.items():
      d[key] = np.array(value)
    return d

  def simple_generator(self, distribution, observation):
    for dist_sample, obs_sample in zip(distribution, observation):
      yield {
        "observation": obs_sample,
        "distribution": dist_sample,
      }

  def test_simple_generator(self):
    test_distribution = np.split(np.random.rand(10, 20, 20), 10)
    test_observation =  np.split(np.random.rand(10, 20, 20), 10)

    test_generator = self.simple_generator(test_distribution, test_observation)

    for c, sample in enumerate(test_generator):
      self.assertAllEqual(test_distribution[c], sample["distribution"])
      self.assertAllEqual(test_observation[c], sample["observation"])

  def test_dataset_from_generator_single_shard(self):
    test_distribution = np.split(
      np.random.rand(10, 20, 20).astype(np.float32), 10)
    test_observation =  np.split(
      np.random.rand(10, 20, 20).astype(np.float32), 10)

    test_generator = self.simple_generator(test_distribution, test_observation)

    observation_spec = defs.ObservationSpec(
      angles=[0, 1., 2.],
      frequencies=[3.e6, 4.e6, 5.e6],
      grid_dimension=.5e-4,
      transducer_bandwidth=1.,
      numerical_aperture=2.,
    )

    generate_records._dataset_from_generator(
      test_generator, observation_spec, self.test_dir, "test_name", 10, 10)

    files = glob.glob(self.test_dir + "/*")
    test_dataset = tf.data.TFRecordDataset(files)
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    next_dist, next_obs, next_obs_params = iterator.get_next()

    with self.test_session() as sess:
      for i  in range(10):
        dist_eval, obs_eval, params_eval = sess.run(
          [next_dist, next_obs, next_obs_params])

        self.assertAllEqual(test_distribution[i], dist_eval)
        self.assertAllEqual(test_observation[i], obs_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)

  @parameterized.expand([
    (50, 10),
    (75, 3),
    (80, 30)
  ])
  def test_dataset_from_generator_multi_shard(
      self, dataset_size, examples_per_shard):
    test_distribution = np.split(
      np.random.rand(dataset_size, 20, 20).astype(np.float32), dataset_size)
    test_observation =  np.split(
      np.random.rand(dataset_size, 20, 20).astype(np.float32), dataset_size)

    test_generator = self.simple_generator(test_distribution, test_observation)

    observation_spec = defs.ObservationSpec(
      angles=[0, 1., 2.],
      frequencies=[3.e6, 4.e6, 5.e6],
      grid_dimension=.5e-4,
      transducer_bandwidth=1.,
      numerical_aperture=2.,
    )

    generate_records._dataset_from_generator(
      test_generator, observation_spec, self.test_dir, "test_name", dataset_size, examples_per_shard)

    files = sorted(glob.glob(self.test_dir + "/*"))
    print(files)
    test_dataset = tf.data.TFRecordDataset(files)
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    next_dist, next_obs, next_obs_params = iterator.get_next()

    with self.test_session() as sess:
      for i  in range(dataset_size):
        dist_eval, obs_eval, params_eval = sess.run(
          [next_dist, next_obs, next_obs_params])

        self.assertAllEqual(test_distribution[i], dist_eval)
        self.assertAllEqual(test_observation[i], obs_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)


if __name__ == "__main__":
  tf.test.main()