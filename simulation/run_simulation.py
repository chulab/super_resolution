"""Main file to run simulation on scatterer dataset.

Example usage:
  python run_simulation.py

"""
import argparse
import glob
import os
import sys
from typing import List, Callable
import multiprocessing as mp
import time

import logging
import multiprocessing_logging

import numpy as np

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import defs
from simulation import create_observation_spec
from simulation import simulate

from training_data import record_writer

from utils import array_utils

def _load_data(
  files: List,
  queue: mp.JoinableQueue,
):
  """Loads elements sequentially from input files.

  Args:
    files: Sorted list of files to be loaded and passed to queue.
    queue: `queue.Queue` in which to put numpy arrays.
  """
  for file in files:
    logging.info("loading file {}".format(file))
    array = np.load(file)
    # `split_array` is a list of individual scatterer distributions.
    split_array = array_utils.reduce_split(array, 0)

    for arr in split_array:
      queue.put(arr.astype(np.float32))


def _files_in_directory(
    directory,
    extension: str='',
):
  files=glob.glob(directory + "/*" + extension)
  return sorted(files)

def _simulate(
    simulate_fn: Callable,
    queue_in: mp.JoinableQueue,
    queue_out: mp.JoinableQueue
):
  """Applies simulate_fn to `input_queue`, places result in `output_queue`."""
  while True:
    try:
      distribution = queue_in.get()
      logging.debug("Simulating")
      time_start = time.time()
      simulation = simulate_fn(distribution[np.newaxis])[0]
      logging.debug(
        "Done simulation took {} sec".format(time.time() - time_start))
      queue_in.task_done()
      queue_out.put((distribution, simulation))
    except Exception:
      logging.error("Fatal error in main loop", exc_info=True)



def _save(
    save_fn,
    queue_in
):
  while True:
    distribution, simulation = queue_in.get()
    logging.info("Saving.")
    save_fn(distribution, simulation)
    queue_in.task_done()


def simulate_and_save(
    scatterer_distribution_directory: str,
    output_directory: str,
    observation_spec: defs.ObservationSpec,
    transverse_psf_dimension: float,
    axial_psf_dimension: float,
    simulation_worker_count: int,
    dataset_name: str,
    examples_per_shard: int,
):
  """Creates dataset by simulating observation of scatterer distribution.

  This function builds a dataset of simulated ultrasound images.

  First, it loads distribution samples from a specified directory.

  Then it simulates US observation parametrized by an `ObservationSpec` using
  (optionally) multiple workers.

  Finally it then saves generated data to disk as a `tfrecords` example.

  Each example contains:
    * Distribution - Array of `[Height, Width]`
    * Observation - Array of `[Height', Width']` (may be different from
      `Distribution`.
    * ObservationSpec - Information on simulation.
    For further information see `record_utils._construct_example`.

  Examples are stored in `output_directory` in a set of files. Each file will
  contain `examples_per_shard` examples.

  Args:
    distributions: `np.ndarray` of shape `[batch, height, width]` describing
      distribution of scatterers.
    output_directory: str,
    observation_spec: `ObservationSpec` parameterizing simulation.
    datset_name: Name describing dataset (e.g. `train`, `eval`).
    output_directory: Path to directory for dataset.
    examples_per_shard: `int` of number of examples per shard (file) in dataset.

  Raises:
    ValueError: If `distributions` does not have shape
      `[num_examples, height, width]`.
  """

  # `scatterer_queue` contains scatterer distributions to be passed to
  # simulation.
  scatterer_queue = mp.JoinableQueue(maxsize=50)

  # `simulated_queue` contains arrays that have already been simulated.
  simulated_queue = mp.JoinableQueue(maxsize=50)

  filenames = _files_in_directory(
    scatterer_distribution_directory, extension=".npy")

  logging.info("Found files {}".format(filenames))

  # Create simulator. This will be copied by each `simulation_worker` process.
  simulator = simulate.USSimulator(
    grid_unit=observation_spec.grid_dimension,
    angles=observation_spec.angles,
    psf_descriptions=observation_spec.psf_descriptions,
    psf_axial_length=axial_psf_dimension,
    psf_transverse_length=transverse_psf_dimension,
  )

  # Create `RecordWriter`. This will be used to write out examples.
  writer = record_writer.RecordWriter(
    directory=output_directory,
    dataset_name=dataset_name,
    examples_per_shard=examples_per_shard,
  )

  # Create loading workers.
  loading_worker = mp.Process(target=_load_data, args=(filenames, scatterer_queue))

  # Create simulation workers.
  simulation_workers = []
  for i in range(simulation_worker_count):
    worker = mp.Process(
      target= _simulate,
      args=(simulator.simulate, scatterer_queue, simulated_queue,)
    )
    worker.name="simulation_worker_{}".format(i)
    logging.debug("Instantiating simulation worker {}".format(worker.name))
    worker.daemon = True
    simulation_workers.append(worker)

  num_saving_threads = 1
  # Create saving workers.
  saving_workers = []
  for i in range(num_saving_threads):
    worker = mp.Process(
      target=_save,
      args=(writer.save, simulated_queue,)
    )
    worker.daemon = True
    saving_workers.append(worker)

  ### LAUNCH WORKERS ###

  # Launch saving workers.
  [worker.start() for worker in saving_workers]

  # Launch simulation threads.
  [worker.start() for worker in simulation_workers]

  # Launch loading threads.
  loading_worker.start()

  print("BEFORE JOIN", flush=True)

  time.sleep(1.)

  scatterer_queue.join()
  simulated_queue.join()

  print("AFTER JOIN", flush=True)

  writer.close()

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output_dir',
                      dest='output_dir',
                      help='Path to save the dataset.',
                      type=str,
                      required=False)

  parser.add_argument('-d', '--distribution',
                      dest='distribution_path',
                      help='Path to the scatterer distribution dataset (np.ndarray).',
                      type=str,
                      required=True)

  parser.add_argument('-w', '--worker_count',
                      dest='worker_count',
                      help='Number of workers to perform simulation.',
                      type=int,
                      required=False)

  parser.add_argument('-os', '--observation_spec_path',
                      dest='observation_spec_path',
                      help='Path to the `observation_spec` param JSON file.',
                      type=str,
                      required=True)

  parser.add_argument('-tpsf', '--transverse_psf_length',
                      dest='transverse_psf_length',
                      help='Length of transverse psf in meters.',
                      type=float,
                      required=True)

  parser.add_argument('-apsf', '--axial_psf_length',
                      dest='axial_psf_length',
                      help='Length of axial psf in meters',
                      type=float,
                      required=True)

  parser.add_argument('-n', '--name', dest='dataset_name',
                      help='Prefix for the tfrecords (e.g. `train`, `test`, `val`).',
                      type=str,
                      default="simulated_us",
                      required=False)

  parser.add_argument('-eps', '--examples_per_shard',
                      dest='examples_per_shard',
                      help='Number of examples per shard.',
                      type=int,
                      default=100,
                      required=False)

  parsed_args = parser.parse_args()

  return parsed_args


def _set_up_logging():
  """Sets up logging."""

  # Check for environmental variable.
  file_location = os.getenv('JOB_DIRECTORY', '.')

  print("Logging file writing to {}".format(file_location), flush=True)

  logging.basicConfig(
    filename=os.path.join(file_location, 'simulation.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(process)d - %(message)s'
  )
  logging.debug("Initialize debug.")
  multiprocessing_logging.install_mp_handler()
  logging.debug("Instantiated `mp_handler` to handle multiprocessing.")


def main():

  _set_up_logging()

  args = parse_args()

  directory=args.distribution_path

  if not os.path.isdir(directory):
    directory = os.path.dirname(directory)

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path)

  simulate_and_save(
    scatterer_distribution_directory=directory,
    output_directory=args.output_dir,
    observation_spec=observation_spec,
    transverse_psf_dimension=args.transverse_psf_length,
    axial_psf_dimension=args.axial_psf_length,
    simulation_worker_count=args.worker_count,
    dataset_name=args.dataset_name,
    examples_per_shard=args.examples_per_shard,
  )


if __name__ == "__main__":
  main()