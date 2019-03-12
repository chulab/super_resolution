"""Creates `ObservationSpec`.

Example usage:
python simulation/create_observation_spec.py -sd /Users/noah/Documents/CHU/super_resolution/super_resolution/simulation/test_data -n test_observation_spec -gd 1e-4 -a 0,.78,1.57 -m 1 -f 2.e6,4e6 -na .125,.125 -fs 5.e5,5.e5
"""

import argparse
import itertools
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
    d = json.load(f)

    return defs.ObservationSpec(
      grid_dimension=d['grid_dimension'],
      angles=d['angles'],
      psf_descriptions=[
        defs.PsfDescription(*description) for description
        in d['psf_descriptions']]
    )


def _generate_psf_description(frequencies, frequency_sigma, modes,
                              numerical_aperture):
  """Returns all cartesian products of frequencies and modes.

  Args:
    frequencies: List of frequencies.
    frequency_sigma: List of same length as `frequencies` containing the
      standard deviation of the gaussian at `frequency`.
    modes: List of gaussian modes.
    numerical_aperture: List of same length as `frequencies` describing NA.

  Returns:
    List of `PsfDescription`
  """
  return [defs.PsfDescription(
    frequency=freq, mode=mode, frequency_sigma=freq_sigma,
    numerical_aperture=na) for (freq, freq_sigma, na), mode in
          itertools.product(
            zip(frequencies, frequency_sigma, numerical_aperture), modes)]


def observation_spec_from_frequencies_and_modes(
    grid_dimension,
    angles,
    frequencies,
    frequency_sigma,
    numerical_aperture,
    modes,
):
  """Generates a `ObservationSpec` given a set of frequencies and modes.

  First builds a set of `PsfDescription` from the cartesian product of
  frequencies and modes. Then constructs `ObservationSpec`

  Args:
    angles: List of angles in radians.
    frequencies: List of frequencies in Hz.
    frequency_sigma: List of frequency bandwidths in same order as
      `frequencies`.
    numerical_aperture: List of NA in same order as `frequencies`.
    modes: List of gaussian modes.

  Returns:
    `ObservationSpec` object.
  """
  if any(len(a) != len(b) for a, b in itertools.combinations(
      [frequencies, frequency_sigma, numerical_aperture], 2)):
    raise ValueError("`frequencies`, `frequency_sigma`, and "
                     "`numerical_aperture` must all have same number of "
                     "elements")

  descriptions = _generate_psf_description(
    frequencies=frequencies,
    frequency_sigma=frequency_sigma,
    modes=modes,
    numerical_aperture=numerical_aperture
  )

  return defs.ObservationSpec(
    grid_dimension=grid_dimension,
    angles=angles,
    psf_descriptions=descriptions,
  )


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

  parser.add_argument('-gd', '--grid_dimension', dest='grid_dimension',
                      help='Grid dimension',
                      type=float,
                      required=True)

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

  parser.add_argument('-na', '--numerical_aperture', dest='numerical_aperture',
                      help='List of numerical aperture',
                      type=lambda s: [float(f) for f in s.split(',')],
                      required=True)

  parser.add_argument('-fs', '--frequency_sigma', dest='frequency_sigma',
                      help='List of frequency sigma',
                      type=lambda s: [float(f) for f in s.split(',')],
                      required=True)

  parsed_args = parser.parse_args()

  return parsed_args


def main():
  args = parse_args()
  observation_spec = observation_spec_from_frequencies_and_modes(
    grid_dimension=args.grid_dimension,
    angles=args.angles,
    frequencies=args.frequencies,
    frequency_sigma=args.frequency_sigma,
    numerical_aperture=args.numerical_aperture,
    modes=args.modes,
  )
  save_observation_spec(observation_spec, args.save_dir, args.name)


if __name__ == "__main__":
  main()