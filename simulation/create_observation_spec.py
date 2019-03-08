"""Creates `ObservationSpec`.

Example usage:

python simulation/create_observation_spec.py -sd /Users/noah/Documents/CHU/super_resolution/super_resolution/simulation/test_data -n test_observation_spec -a 0,1.04,2.09 -m 0,1 -f 2e6,5e6,7e6 -gd 1e-4 -tb .1 -na .125
"""

import argparse
import json
import os
import sys

# Add `super_resolution` package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from simulation import defs

def save_observation_spec(
    observation_spec: defs.ObservationSpec,
    save_dir: str,
    name: str = "observation_spec"
):
  file_name = os.path.join(save_dir, name + ".json")

  with open(file_name, "w") as file:
    file.write(json.dumps(observation_spec._asdict()))


def load_observation_spec(
    file_path: str
):
  """Loads `ObservationSpec` from file path."""
  with open(file_path, 'r') as f:
    return defs.ObservationSpec(**json.load(f))


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('-sd', '--save_dir', dest='save_dir',
                      help='Path to save the ObservationSpec.',
                      type=str,
                      required=True)

  parser.add_argument('-n', '--name', dest='name',
                      help='Optional name.',
                      type=str,
                      required=False)

  parser.add_argument('-a', '--angles', dest='angles',
                      help='Comma delimited list of angles',
                      type=lambda s: [float(angle) for angle in s.split(',')],
                      required=True)

  parser.add_argument('-m', '--modes', dest='modes',
                      help='Comma delimited list of modes',
                      type=lambda s: [int(mode) for mode in s.split(',')],
                      required=True)

  parser.add_argument('-f', '--frequencies', dest='frequencies',
                      help='Comma delimited list of frequencies',
                      type=lambda s: [float(f) for f in s.split(',')],
                      required=True)

  parser.add_argument('-gd', '--grid_dimension', dest='grid_dimension',
                      help='Grid dimension',
                      type=float,
                      required=True)

  parser.add_argument('-tb', '--transducer_bandwidth', dest='transducer_bandwidth',
                      help='Transducer bandwidth.',
                      type=float,
                      required=True)

  parser.add_argument('-na', '--numerical_aperture', dest='numerical_aperture',
                      help='numerical aperture',
                      type=float,
                      required=True)

  parsed_args = parser.parse_args()

  return parsed_args


def main():
  args = parse_args()
  observation_spec = defs.ObservationSpec(
    angles=args.angles,
    frequencies=args.frequencies,
    modes=args.modes,
    grid_dimension=args.grid_dimension,
    transducer_bandwidth=args.transducer_bandwidth,
    numerical_aperture=args.numerical_aperture
  )
  save_observation_spec(observation_spec, args.save_dir, args.name)


if __name__ == "__main__":
  main()