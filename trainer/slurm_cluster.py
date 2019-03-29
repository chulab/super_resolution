"""Util for getting `cluster_spec` when running on SLURM cluster."""

import argparse
import logging

import tensorflow as tf


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--ps_count',
    type=int,
    required=True,
  )

  parser.add_argument(
    '--worker_count',
    type=int,
    required=True,
  )

  parser.add_argument(
    '--gpus_per_node',
    type=int,
    required=True,
  )

  parser.add_argument(
    '--gpus_per_task',
    type=int,
    required=False,
    default=1,
  )

  parser.add_argument(
    '--port_base',
    type=int,
    required=False,
    default=8888,
  )

  args, _ = parser.parse_known_args()

  return args


def get_cluster():
  """Returns `ClusterSpec` object to perform distributed training."""

  args = parse_args()

  cluster_resolver = tf.contrib.cluster_resolver.SlurmClusterResolver(
      {'ps': args.ps_count, 'worker': args.worker_count},
      port_base=args.port_base,
      gpus_per_node=args.gpus_per_node,
      gpus_per_task=args.gpus_per_task,
      auto_set_gpu=True,
  )

  logging.info("Set up `cluster_resolver`.")
  logging.info("Found accelerators {}".format(cluster_resolver.num_accelerators))

  return cluster_resolver.cluster_spec()