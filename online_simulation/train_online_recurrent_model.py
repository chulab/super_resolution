"""Main module for training angle-only model."""

import argparse
import logging
import os
import sys
import tensorflow as tf

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from analysis import plot_utils, recorder_utils, summaries
from cloud_utils import save_utils
from oauth2client.client import GoogleCredentials

from cloud_utils import save_utils
from online_simulation import recurrent_model
from online_simulation import online_dataset_utils
from online_simulation import online_simulation_utils

from utils import logging_utils

_PREDICT="PREDICT"
_TRAIN="TRAIN"


def make_train_params():
  return tf.contrib.training.HParams(
    train_steps=2,
    eval_steps=1,
    profile_steps=1,
    save_checkpoint_steps=200,
    log_step_count=20,
  )


def train_and_evaluate(
    output_directory,
    train_input,
    eval_input,
    train_steps,
    eval_steps,
    model_fn,
    model_params,
    warm_start_from,
    profile_steps,
    save_checkpoint_steps,
    log_step_count,
):
  """Run the training and evaluate using the high level API."""
  tf_config = os.environ.get('TF_CONFIG')
  logging.info("TF_CONFIG {}".format(tf_config))

  # Load `RunConfig`.
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(
    model_dir=output_directory,
    log_step_count_steps=log_step_count,
    save_checkpoints_steps=save_checkpoint_steps
  )

  estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config,
    params=model_params,
    warm_start_from=warm_start_from,
  )

  # Hook to log step timing.
  hook = tf.train.ProfilerHook(
    save_steps=profile_steps,
    output_dir=output_directory,
    show_memory=True
  )

  logging.info("Defining `train_spec`.")
  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input,
    max_steps=train_steps,
    hooks=[hook]
  )

  logging.info("Defining `eval_spec`.")
  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input,
    steps=eval_steps,
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--mode',
    help='Either `train` or `predict`',
    default=_TRAIN
  )

  parser.add_argument(
    '--job-dir',
    help='Path to save test_output of model including checkpoints.',
    type=str,
    default='.',
  )

  parser.add_argument(
    '--dataset_params',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
  )

  parser.add_argument(
    '--model_params',
    type=str,
    help='Comma separated list of "name=value" pairs.',
    default=''
  )

  parser.add_argument(
    '--simulation_params',
    help = 'Comma separated list of "name=value" pairs.',
    default = '',
  )

  parser.add_argument(
    '--train_params',
    help = 'Comma separated list of "name=value" pairs.',
    default = '',
  )

  parser.add_argument(
    '--warm_start_from',
    help = 'Warm start file.',
    default = None,
  )

  ## HPARAMETER TUNING ARGS
  parser.add_argument(
    '--angle_count',
    type=int,
    default=1,
  )
  parser.add_argument(
    '--angle_limit',
    type=float,
    default=90.
  )
  parser.add_argument(
    '--frequency_count',
    type=int,
    default=1
  )
  parser.add_argument(
    '--min_frequency',
    type=float,
    default=10e6
  )
  parser.add_argument(
    '--max_frequency',
    type=float,
    default=20e6
  )
  parser.add_argument(
    '--mode_count',
    type=int,
    default=1
  )
  parser.add_argument(
    '--frequency_sigma',
    type=float,
    default=1e6
  )
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=.001,
  )
  parser.add_argument(
    '--train_steps',
    type=float,
    default=2000,
  )

  parser.add_argument(
    '--service_account_path',
    type=str,
    help='path to service account credentials for Google utilities',
    default='gs://chu_super_resolution_data/service-account.json'
  )

  parser.add_argument(
    '--slide_id',
    type=str,
    help='Google slide id for summary',
    default='187e6QMPZApaDQoqC5v_BqQ4Y0mZUsUD1VkviSu_hCnY'
  )

  args, _ = parser.parse_known_args()

  return args


def main():

  args = parse_args()

  logging_utils.set_up_logging()

  dataset_params = online_dataset_utils.dataset_params()
  dataset_params.parse(args.dataset_params)

  simulation_params = online_simulation_utils.simulation_params()
  simulation_params.parse(args.simulation_params)

  descriptions=online_simulation_utils.grid_psf_descriptions(
    angle_limit=args.angle_limit,
    angle_count=args.angle_count,
    min_frequency=args.min_frequency,
    max_frequency=args.max_frequency,
    frequency_count=args.frequency_count,
    mode_count=args.mode_count,
  )
  simulation_params.psf_descriptions=descriptions
  simulation_params.frequency_sigma=args.frequency_sigma

  model_params = recurrent_model.make_hparams()
  model_params.learning_rate=args.learning_rate
  model_params.parse(args.model_params)
  model_params.add_hparam("job_dir", args.job_dir)

  train_params = make_train_params()
  train_params.train_steps=args.train_steps
  train_params.parse(args.train_params)

  logging.info("Model params {}".format(model_params))
  logging.info("Data params {}".format(dataset_params))
  logging.info("Simulation params {}".format(simulation_params))
  logging.info("Train params{}".format(train_params))

  # train_input_fn =lambda: online_dataset_utils.distribution_dataset(
  #   dataset_type=dataset_params.dataset_type,
  #   physical_dimension=dataset_params.physical_dimension,
  #   grid_dimension=dataset_params.grid_dimension,
  #   min_radius=dataset_params.min_radius,
  #   max_radius=dataset_params.max_radius,
  #   max_count=dataset_params.max_count,
  #   background_noise=dataset_params.background_noise,
  #   lambda_multiplier=dataset_params.lambda_multiplier,
  # )
  #
  # psfs = online_simulation_utils.make_psf(
  #   psf_dimension=simulation_params.psf_dimension,
  #   grid_dimension=dataset_params.grid_dimension,
  #   descriptions=simulation_params.psf_descriptions,
  #   numerical_aperture=simulation_params.numerical_aperture,
  #   frequency_sigma=simulation_params.frequency_sigma,
  # )
  #
  # fig = plot_utils.plot_grid(psfs, scale=dataset_params.grid_dimension,)
  # save_utils.maybe_save_cloud(fig, args.job_dir)
  #
  # model_params.psfs=psfs

  if "recurrent" in model_params.recurrent:
    model_fn = recurrent_model.model_fn_recurrent
  else:
    model_fn = recurrent_model.model_fn

  if args.mode==_TRAIN:
    # train_and_evaluate(
    #   output_directory=args.job_dir,
    #   train_input=train_input_fn,
    #   eval_input=train_input_fn,
    #   train_steps=train_params.train_steps,
    #   eval_steps=train_params.eval_steps,
    #   model_fn=model_fn,
    #   model_params=model_params,
    #   warm_start_from=args.warm_start_from,
    #   profile_steps=train_params.profile_steps,
    #   save_checkpoint_steps=train_params.save_checkpoint_steps,
    #   log_step_count=train_params.log_step_count
    # )

    tb_dir = args.job_dir + "/eval"
    save_dir = args.job_dir
    # if args.cloud_train:
    #   save_dir = args.job_dir
    # else:
    #   save_dir = "gs://chu_super_resolution_experiment/test_output"

    steps, _ = plot_utils.get_all_tensor_from_tensorboard(tb_dir, 'predictions/distribution_tensor')
    steps = sorted(list(dict.fromkeys(steps)))
    summaries.summarize_template_3(args, dataset_params, simulation_params,
      model_params, steps, tb_dir, save_dir)



if __name__ == "__main__":
  main()