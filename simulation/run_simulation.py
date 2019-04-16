"""Main file to run simulation on scatterer dataset.

Example usage:
python simulation/run_simulation.py -o /Users/noah/Documents/CHU/super_resolution/super_resolution/simulation/test_data -d /Users/noah/Documents/CHU/super_resolution/super_resolution/simulation/test_data -w 2 -os /Users/noah/Documents/CHU/super_resolution/super_resolution/simulation/test_data/test_observation_spec.json -tpsf 6e-3 -apsf 4e-3 -n test_dataset -eps 2"""
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

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib_scalebar import scalebar

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import defs
from simulation import create_observation_spec
from simulation import simulate

from training_data import record_writer

from utils import array_utils


def _load_or_make_finished_files(file_path):
  if os.path.isfile(file_path):
    with open(file_path) as f:
      return f.read().splitlines()
  else:
    if not os.path.exists(os.path.dirname(file_path)):
      os.makedirs(os.path.dirname(file_path))
    # Touch file.
    open(file_path, 'a').close()
    return []


def _remove_processed(
    filenames, finished_list
):
  """Removes files from `filenames` which are present in `finished_list`"""
  return [file for file in filenames if file not in finished_list]


def _add_file_to_log(
    file, log_file,
):
  with open(log_file) as f:
    f.write(file)
    f.write("\n")


def _files_in_directory(
    directory,
    extension,
    finished_files: List,
):
  files=glob.glob(directory + "/*" + extension)
  files=sorted(files)
  # Remove already processed files.
  filenames = _remove_processed(files, finished_files)
  return filenames


def _load_data(
  files: List,
  queue: mp.JoinableQueue,
  log_file: str
):
  """Loads elements sequentially from input files.

  Args:
    files: Sorted list of files to be loaded and passed to queue.
    queue: `queue.Queue` in which to put numpy arrays.
  """
  try:
    for file in files:
      logging.info("loading file {}".format(file))
      array = np.load(file)
      # `split_array` is a list of individual scatterer distributions.
      split_array = array_utils.reduce_split(array, 0)

      for arr in split_array:
        queue.put(arr.astype(np.float32))
        logging.debug("Put distribution in queue")

      _add_file_to_log(file, log_file)
  except Exception:
    logging.error("Fatal error in loading loop", exc_info=True)


def _simulate(
    simulate_fn: Callable,
    queue_in: mp.Manager().Queue,
    queue_out: mp.Manager().Queue
):
  """Applies simulate_fn to `input_queue`, places result in `output_queue`."""
  while True:
    try:
      distribution = queue_in.get()
      logging.debug("Starting simulation.")
      time_start = time.time()
      simulation = simulate_fn(distribution[np.newaxis])[0]
      logging.debug(
        "Done simulation took {} sec".format(time.time() - time_start))
      queue_out.put((distribution, simulation))
      logging.debug("Put array in `output_queue`")
      queue_in.task_done()
    except Exception:
      logging.error("Fatal error in simulation loop", exc_info=True)
      break


def _save(
    record_writer,
    queue_in: mp.Manager().Queue,
):
  while True:
    try:
      logging.info("Loading arrays to save.")
      distribution, simulation = queue_in.get()
      if distribution is not None:
        logging.info("Starting saving.")
        time_start=time.time()
        record_writer.save(distribution, simulation)
        logging.debug(
          "Done saving took {} sec".format(time.time() - time_start))
      else:
        logging.info("Shutdown signal recieved.")
        record_writer.close()
        logging.info("Closed `record_writer`.")
      queue_in.task_done()
    except Exception:
      logging.error("Fatal error in save", exc_info=True)
      break


def _save_psf(
    psfs,
    descriptions: defs.PsfDescription,
    observation_spec: defs.ObservationSpec,
    file_directory,
):
  for psf, description in zip(psfs, descriptions):
    observation_file_name = os.path.join(
      file_directory, "psf_freq_{}_mode_{}.png".format(
        description.frequency, description.mode))
    fig, ax = plt.subplots(1, 1)
    i = ax.imshow(psf, interpolation=None)

    plt.colorbar(i, ax=ax)

    fig.suptitle(
      "Frequency {freq} Mode {mode} with {pix} pixels.".format(
      freq=description.frequency, mode=description.mode, pix=str(psf.shape)
      ))

    sb = scalebar.ScaleBar(observation_spec.grid_dimension)
    ax.add_artist(sb)

    plt.savefig(observation_file_name)
    plt.close(fig)


def simulate_and_save(
    scatterer_distribution_directory: str,
    output_directory: str,
    observation_spec: defs.ObservationSpec,
    transverse_psf_dimension: float,
    axial_psf_dimension: float,
    simulation_worker_count: int,
    dataset_name: str,
    examples_per_shard: int,
    finished_log: str,
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
    finished_log: Path to a text file where this program will record files that
    have been processed (used for recovery).

  Raises:
    ValueError: If `distributions` does not have shape
      `[num_examples, height, width]`.
  """
  manager = mp.Manager()

  # `scatterer_queue` contains scatterer distributions to be passed to
  # simulation.
  scatterer_queue = manager.Queue(maxsize=30)

  # `simulated_queue` contains arrays that have already been simulated.
  simulated_queue = manager.Queue(maxsize=30)

  finished_files = _load_or_make_finished_files(finished_log)

  filenames = _files_in_directory(
    scatterer_distribution_directory, extension=".npy", finished_files=finished_files)

  logging.info("Found files {}".format(filenames))

  # Create simulator. This will be copied by each `simulation_worker` process.
  simulator = simulate.USSimulator(
    grid_unit=observation_spec.grid_dimension,
    angles=observation_spec.angles,
    psf_descriptions=observation_spec.psf_descriptions,
    psf_axial_length=axial_psf_dimension,
    psf_transverse_length=transverse_psf_dimension,
  )

  # Save psfs.
  _save_psf(simulator.psf, simulator.psf_descriptions, observation_spec, output_directory)
  logging.debug("Saved `.png` of psfs.")

  # Create `RecordWriter`. This will be used to write out examples.
  writer = record_writer.RecordWriter(
    directory=output_directory,
    dataset_name=dataset_name,
    examples_per_shard=examples_per_shard,
  )

  # Create loading workers.
  loading_worker_count = 1
  loading_workers = []
  for i in range(loading_worker_count):
    worker = mp.Process(
      target=_load_data, args=(filenames, scatterer_queue, finished_log))
    worker.name = "loading_worker_{}".format(i)
    worker.daemon=True
    logging.debug("Instantiating loading worker {}".format(worker.name))
    loading_workers.append(worker)

  # Create simulation workers.
  simulation_workers = []
  for i in range(simulation_worker_count):
    worker = mp.Process(
      target= _simulate,
      args=(simulator.simulate, scatterer_queue, simulated_queue,),
    )
    worker.name="simulation_worker_{}".format(i)
    worker.daemon=True
    logging.debug("Instantiating simulation worker {}".format(worker.name))
    simulation_workers.append(worker)

  num_saving_threads = 1
  # Create saving workers.
  saving_workers = []
  for i in range(num_saving_threads):
    worker = mp.Process(
      target=_save,
      args=(writer, simulated_queue,)
    )
    worker.name="saving_worker_{}".format(i)
    worker.daemon=True
    logging.debug("Instantiating saving worker {}".format(worker.name))
    saving_workers.append(worker)

  ### LAUNCH WORKERS ###

  # Launch saving workers.
  [worker.start() for worker in saving_workers]
  logging.debug("Started `saving_workers`.")

  # Launch simulation threads.
  [worker.start() for worker in simulation_workers]
  logging.debug("Started `simulation_workers`.")

  # Launch loading threads.
  [worker.start() for worker in loading_workers]
  logging.debug("Started `loading_workers`.")

  [worker.join() for worker in loading_workers]
  logging.debug("Joined `loading_workers`.")

  scatterer_queue.join()
  logging.debug("Joined `scatterer_queue`.")

  [worker.terminate() for worker in simulation_workers]
  logging.debug("Closed `simulation_workers`.")

  simulated_queue.put((None, None))
  logging.debug("Placed shutdown object in `simulated_queue`.")

  simulated_queue.join()
  logging.debug("Joined `simulated_queues`.")

  [worker.terminate() for worker in saving_workers]
  logging.debug("Closed `saving_workers`.")


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


def _log_system_stats():
  """Logs basic system stats"""

  # CPU information.
  try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    logging.debug("Found `SLURM_JOB_CPUS_PER_NODE`")
  except KeyError:
    ncpus = mp.cpu_count()
    logging.debug("Using `mp.cpu_count`")
  logging.debug("CPU COUNT {}".format(ncpus))

  # Memory information.
  try:
    mem_per_node = int(os.environ["SLURM_MEM_PER_NODE"])
    logging.debug("Found `SLURM_MEM_PER_NODE`")
    logging.debug("MEMORY PER CPU CORE {}".format(mem_per_node / ncpus))
  except KeyError:
    logging.debug("Memory information not available.")

  # Node information
  try:
    node_list = int(os.environ["SLURM_JOB_NODELIST"])
    logging.debug("Found `SLURM_JOB_NODELIST` \n Nodelist {}".format(node_list))
  except:
    logging.debug("`SLURM_JOB_NODELIST` not found.")


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
  _log_system_stats()

  args = parse_args()

  directory=args.distribution_path

  if not os.path.isdir(directory):
    directory = os.path.dirname(directory)

  finished_log=os.path.join(directory, "finished_log")
  logging.info("using `finished_log` {}".format(finished_log))

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
    finished_log=finished_log,
  )


if __name__ == "__main__":
  main()