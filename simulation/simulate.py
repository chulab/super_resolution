"""Defines the `USSimulator` - highest level interface to simulation code.

Example Usage:
  #  US = USSimulator()
  #  state = ...
  #  simulated_images = SSimulator.simulate(state)
"""
import itertools
from typing import List

import numpy as np

from simulation import psf_utils
from simulation import response_functions
from simulation import defs
from simulation import observation


class USSimulator(object):
  """Runs ulstrasound simulation.

  This function is the highest level interface with the ultrasound simulation
  framework. It is used to simulate arbitrary number of US images using
  arbitrary parameters (angles, modes, frequencies).

  Each simulation uses the same PSF and simulation parameters (i.e. the same
  angles, modes, and frequencies).

  It is necessary to set the `frequencies` and `modes` before doing simulation
  as these parametrize the impulse response functions that are used for
  simulation.

  Attributes:
    grid_unit: Float describing grid size in meters.
    angles: List of angles from which to perform simulation.
    psf_description: List of `PsfDescription` used to construct PSF for
      simulation.
    psf_axial_length: Float describing physical size of psf used for simulation
      in meters.
    psf_transverse_length: Same as `psf_axial_length` but for transverse
      direction.

  """

  def __init__(
      self,
      grid_unit: float,
      angles: List[float],
      psf_descriptions: List[defs.PsfDescription],
      psf_axial_length: float,
      psf_transverse_length: float,
  ):
    self._grid_unit = grid_unit

    self._angles = angles

    self._psf_descriptions = psf_descriptions

    self.psf_physical_size = [psf_transverse_length, 0., psf_axial_length]

    # Note that the `z` or `axial` sampling of the psf is TWICE that of the
    # lateral and axial samplings. This takes into account the round trip time
    # experienced by the pulse as it reflects.
    psf_grid_dimensions = [
      self._grid_unit, self._grid_unit, self._grid_unit * 2]

    self._psf_coordinates = self._coordinates(self.psf_physical_size,
                                              psf_grid_dimensions)

    # Generate the impulse used for simulation.
    self._psf = self._build_psf(
      coordinates=self._psf_coordinates,
      psf_descriptions=self._psf_descriptions,
    )

    # Convert list of psf's into filter with shape
    # `[height, width, 1, psf_count]`.
    self._impulse_filter = psf_utils.to_filter(self._psf, mode="FROM_SINGLE")

  def _coordinates(self, lengths, grid_dimensions):
    """Makes coordinate grid for impulse response."""
    xx, yy, zz = response_functions.coordinate_grid(
      lengths, grid_dimensions, center=True)
    return np.stack([xx, yy, zz], -1)

  def _build_psf(
      self,
      coordinates,
      psf_descriptions,
  ):
    """Returns list of psfs based on `psf_descriptions`."""
    psfs = []
    for description in psf_descriptions:
      # Generate psf from each description.
      psf_temp = response_functions.gaussian_impulse_response_v2(
        coordinates=coordinates,
        frequency=description.frequency,
        mode=description.mode,
        numerical_aperture=description.numerical_aperture,
        frequency_sigma=description.frequency_sigma,
      )[:, 0, :]

      # Swap `x` and `z` axes.
      psf_temp = np.transpose(psf_temp, [1, 0])
      psfs.append(psf_temp.astype(np.float32))
    return psfs

  @property
  def psf(self):
    return self._psf

  @property
  def psf_descriptions(self):
    return self._psf_descriptions

  @property
  def angles(self):
    return self._angles

  @angles.setter
  def angles(self, angles):
    self._angles = angles

  @property
  def grid_unit(self):
    """Returns response function grid units.

    Note that this is different than the response function grid units, as the
    axial grid size is TWICE that of the transverse units.
    """
    return self._grid_unit

  def simulate(
      self,
      scatter_distribution,
  ):
    """Simulates US image of `scatter_distribution`.

    args:
      scatter_distribution: distribution of shape `[batch, height, width]`.

    returns:
      Batch of simulated Ultrasound image.
    """
    return self._simulate(scatter_distribution)

  def _simulate(self, scatter_distribution):
    return observation.rotate_and_observe_np(
      scatter_distribution,
      self._angles,
      self._impulse_filter,
    )