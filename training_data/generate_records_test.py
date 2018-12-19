"""Tests for `generate_records.py`"""
import glob
import shutil
import tempfile
from parameterized import parameterized

import numpy as np
import tensorflow as tf

from simulation import defs
from simulation import defs_test
from simulation import estimator
from training_data import generate_records
from training_data import record_utils
from training_data import dataset_utils


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

  def simple_generator(self, distribution, observation, batch_size):
    def chunks(l, n):
      """Yield successive n-sized chunks from l."""
      for i in range(0, len(l), n):
        yield l[i:i + n]

    for dist_sample, obs_sample in zip(
        chunks(distribution, batch_size),
        chunks(observation, batch_size)
    ):
      yield {
        "observation": np.concatenate(obs_sample, axis=0),
        "distribution": np.concatenate(dist_sample, axis = 0),
      }

  def test_simple_generator(self):
    test_distribution = np.split(np.random.rand(10, 20, 20), 10)
    test_observation =  np.split(np.random.rand(10, 20, 20), 10)

    test_generator = self.simple_generator(
      test_distribution, test_observation, 1)

    for c, sample in enumerate(test_generator):
      self.assertAllEqual(test_distribution[c], sample["distribution"])
      self.assertAllEqual(test_observation[c], sample["observation"])

  def testReduceSplit(self):
    array = np.array([[1, 2], [3.4, 5.7], [2.1, 4.5]])
    splits = generate_records.reduce_split(array, 0)
    true_splits = [np.array([1, 2]), np.array([3.4, 5.7]),
                   np.array([2.1, 4.5])]
    for t, s in zip(true_splits, splits):
      self.assertAllEqual(t, s)


  def test_dataset_from_generator_single_shard(self):
    test_distribution = np.split(
      np.random.rand(10, 20, 25).astype(np.float32), 10)
    test_observation =  np.split(
      np.random.rand(10, 30, 35).astype(np.float32), 10)

    test_generator = self.simple_generator(
      test_distribution, test_observation, 1)

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

        self.assertAllEqual(dist_eval.shape, (20, 25))
        self.assertAllEqual(obs_eval.shape, (30, 35))

        self.assertAllEqual(test_distribution[i][0], dist_eval)
        self.assertAllEqual(test_observation[i][0], obs_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)

  @parameterized.expand([
    (50, 10),
    (75, 3),
    (80, 30)
  ])
  def test_dataset_from_generator_multi_shard(
      self, dataset_size, examples_per_shard):
    test_distribution = np.split(
      np.random.rand(dataset_size, 20, 25).astype(np.float32), dataset_size)
    test_observation =  np.split(
      np.random.rand(dataset_size, 30, 35).astype(np.float32), dataset_size)

    test_generator = self.simple_generator(
      test_distribution, test_observation, 1)

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
    test_dataset = tf.data.TFRecordDataset(files)
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    next_dist, next_obs, next_obs_params = iterator.get_next()

    with self.test_session() as sess:
      for i  in range(dataset_size):
        dist_eval, obs_eval, params_eval = sess.run(
          [next_dist, next_obs, next_obs_params])

        self.assertAllEqual(test_distribution[i][0], dist_eval)
        self.assertAllEqual(test_observation[i][0], obs_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)

  @parameterized.expand([
    ([10] * 1,),
    ([10] * 2,),
    ([10] * 4,)
  ])
  def testCreateBadDistributions(self, shape):
    distributions = np.random.rand(*shape)
    observation_spec = defs_test.simple_observation_spec()
    axial_psf_length = 5
    lateral_psf_length = 5
    dataset_name = "test_name"
    output_directory = self.test_dir
    examples_per_shard = 10

    with self.assertRaisesRegex(ValueError, "`distributions` must have shape"):
      generate_records.create(
        distributions, observation_spec, axial_psf_length, lateral_psf_length,
        dataset_name, output_directory, examples_per_shard)

  def testCreate(self):
    """Tests datset created by `create` against simulation.

    This test checks the operation of `create` by doing the following:

    // Generate dataset using `create`.
    //
    //  For each `Example` in dataset:
    //  * Checking that a manual simulation on `distribution` gives the
    //  corresponding `observation`.
    //  * Checking that each example `distribtution` matches the distributions
    //    used to initialize `create`.
    //  * Checks that saved `observationSpec` corresponds to one used to
    //    generate dataset.
    """
    dataset_size = 50
    distributions = np.random.rand(dataset_size, 10, 10).astype(np.float32)
    observation_spec = defs_test.simple_observation_spec()
    axial_psf_length = 5
    lateral_psf_length = 5
    dataset_name = "test_name"
    output_directory = self.test_dir
    examples_per_shard = 50

    # Write records using `create`.
    generate_records.create(
      distributions, observation_spec, axial_psf_length, lateral_psf_length,
      dataset_name, output_directory, examples_per_shard)

    files = sorted(glob.glob(self.test_dir + "/*"))
    test_dataset = tf.data.TFRecordDataset(files)
    test_dataset = test_dataset.map(record_utils._parse_example)
    iterator = test_dataset.make_one_shot_iterator()
    saved_dist, saved_obs, saved_obs_params = iterator.get_next()

    obs_eval_ = []
    with self.test_session() as sess:
      for i  in range(dataset_size):
        dist_eval, obs_eval, params_eval = sess.run(
          [saved_dist, saved_obs, saved_obs_params])

        obs_eval_.append(obs_eval)

        self.assertAllEqual(distributions[i], dist_eval)
        self.assertAllClose(observation_spec._asdict(), params_eval)

    test_simulation_estimator = estimator.SimulationEstimator(
      observation_spec, axial_psf_length, lateral_psf_length
    )
    test_obs = test_simulation_estimator.predict(
      lambda: dataset_utils.array_input_fn(distributions, "PREDICT", 1),
      yield_single_examples=False,
    )

    for sim_obs, saved_obs in zip(next(test_obs)["observation"], obs_eval_):
      self.assertAllEqual(sim_obs, saved_obs)


if __name__ == "__main__":
  tf.test.main()