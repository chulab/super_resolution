"""Main file for scatterer distribution construction.

Example usage:
# generate_scatterer_dataset \
# -o './test_dir'
# -n 'test_name'
# -t 'CIRCLE'
# -s 10. 10.
# -gd 1. 1.
# -eps 100
# -c 1000
# -l 1000
#
# --min_radius 1
# --max_radius 4
# --max_count 4
# --background_noise .05
"""

import argparse
import os
import sys

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_data import shapes_dataset
from training_data import particle_dataset
from training_data import numpy_dataset_utils

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('-o', '--output_dir', dest='output_dir',
                      help='Path to save the dataset.',
                      type=str,
                      required=True)

  parser.add_argument('-n', '--name', dest='dataset_name',
                      help='Prefix for the dataset (e.g. `train`, `test`, `val`).',
                      type=str,
                      required=True)

  parser.add_argument('-t', '--type', dest='distribution_type',
                      help='Type of scatterer to simulate. Valid modes: ',
                      type=str, required=True)

  parser.add_argument('-s', '--size', dest='size', type=float, nargs=2,
                      required=True)

  parser.add_argument('-gd', '--grid_dimension', dest='grid_dimension',
                      type=float, nargs=2, required=True)

  parser.add_argument('-eps', '--examples_per_shard', dest='examples_per_shard',
                      help='Number of shards to make.', type=int,
                      required=True)

  parser.add_argument('-c', '--count', dest='count',
                      help='Number of distributions to generate.',
                      type=int, required=True)

  parser.add_argument('-l', '--lambda', dest='lambda_multiplier',
                      help='Lambda multiplier. See documentation for'
                           '`particle_dataset.poisson_noise`.',
                      type=int, required=True)

  args, unknown = parser.parse_known_args()

  return args


def main():
  args = parse_args()

  # Make distribution generator.
  distribution_generator = shapes_dataset.generate_shapes(
    type=args.distribution_type,
    physical_size=args.size,
    grid_dimensions=args.grid_dimension
  )

  # Add poissonian noise.
  poisson_generator = particle_dataset.poisson_generator(
    distribution_generator, lambda_multiplier=args.lambda_multiplier,
    normalize_output=True)

  # Save to disk.
  numpy_dataset_utils.save_dataset(
    generator=poisson_generator,
    count=args.count,
    shard_size=args.examples_per_shard,
    save_directory=args.output_dir,
    file_prefix=args.dataset_name,
  )


if __name__ == "__main__":
  main()