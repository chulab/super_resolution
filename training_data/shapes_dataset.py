"""Functions to generate dataset (in shards) for certain geometric shapes"""

import numpy as np
from math import tan, pi, ceil, floor
import random
from skimage.draw import line_aa

VALID_TYPES = ['circle', 'line']

def draw_circle(
    array_size: int,
    origin,
    radius: int,
    num_dim: int,
):
    """Returns np.ndarray with a circle of given radius at a center.
    Set cells are `1` while empty cells are `0`.

    Args:
        array_size: size of each dimension of array.
        origin: indices of circle's center in array as a Tuple of ints with
                length equal to num_dim.
        radius: radius of circle in terms of length along array.
        num_dim: number of dimensions of array.

    Returns:
        np.ndarray with total dimensions num_dim with each dimension having
        size `array_size`.

    Raises:
      ValueError: if len of origin tuple is not num_dim.
    """

    if len(origin) != num_dim:
      raise ValueError("`origin` must be a tuple of `num_dim` ints."
                        " Got tuple of %d ints but `num_dim` was %d "
                        % (len(origin), num_dim))

    # equivalent to [0:array_size, 0:array_size, ...] repeated num_dim times
    args = [slice(i,j) for i,j in [(0, array_size)] * num_dim]

    # all_coordinates[i] contains the i-coordinate of each array element
    all_coordinates = np.mgrid[args]

    # compute square distances of each array element from origin
    distances = np.zeros([array_size] * num_dim)
    for i in range(num_dim):
        distances += (all_coordinates[i] - origin[i]) ** 2

    circle = (distances <= radius * radius).astype(int)
    return circle

def circle_fn(
    physical_dim: float,
    grid_size: float,
    min_rad: float,
    max_rad: float,
    max_count: int,
    num_dim: int = 2,
):
    """Produces a box of up to max_count circles with radii between min_rad
    and max_rad, represented by a np.ndarray.

    Set cells are `1` while empty cells are `0`.

    Args:
      physical_dim: length of box in metres.
      grid_size: grid size in metres.
      min_rad: minimum circle radius in metres.
      max_rad: maximum circle radius in metres.
      max_count: maximum number of circles generated.
      num_dim: number of dimensions of box (2 or 3).

    Returns:
      np.ndarray with total dimensions num_dim with each dimension having
      size `physical_dim / grid_size`.
    """

    array_size = ceil(physical_dim / grid_size)
    min_array_rad = floor(min_rad / grid_size)
    max_array_rad = ceil(max_rad / grid_size)
    count = np.random.randint(1, max_count + 1)

    box = np.zeros([array_size] * num_dim)
    for _ in range(count):

        # get circle parameters
        origin = np.random.randint(0, array_size, num_dim)
        radius = np.random.randint(min_array_rad, max_array_rad + 1)
        circle = draw_circle(array_size, origin, radius, num_dim)

        #set circle
        box = np.logical_or(box, circle)

    return box.astype(int)

def line_2d_endpoints(array_size: int, origin, grad: float):
    """Returns endpoints of the 2d line passing through an origin with a given
    gradient in an array.

    Args:
        array_size: size of each dimension of array.
        origin: Tuple(int, int) indices of a point on the line.
        grad: gradient at origin.

    Returns:
        Tuple(int, int, int, int) representing coordinates of left and right
        endpoints in the form of (x_left, y_left, x_right, y_right).
    """

    x_left = y_left = x_right = y_right = 0

    if grad == 0:
        return 0, origin[1], array_size, origin[1]
    elif grad > 0:
        x_diff_left = max(-1 * origin[0] , -1 * float(origin[1]) / grad)
        x_diff_right = min(array_size - 1 - origin[0], \
            float(array_size - 1 - origin[1]) / grad)
    else:
        x_diff_left = max(-1 * origin[0], \
            -1 * float(origin[1] - array_size + 1) / grad)
        x_diff_right = min(array_size - 1 - origin[0], \
            - 1 * float(origin[1]) / grad)

    x_left = floor(origin[0] + x_diff_left)
    y_left = ceil(origin[1] + x_diff_left * grad)
    x_right = floor(origin[0] + x_diff_right)
    y_right = ceil(origin[1] + x_diff_right * grad)

    return x_left, y_left, x_right, y_right

def line_fn(
    physical_dim: float,
    grid_size: float,
    max_count: int,
    sharpness: int = 100,
    num_dim: int = 2,
):
    """Produces a box of up to max_count lines, represented by a np.ndarray.

    Set cells are `1` while empty cells are `0`.
    Increasing sharpness causes line to be thinner and more salient.

    Args:
      physical_dim: length of box in metres.
      grid_size: grid size in metres.
      max_count: maximum number of lines generated.
      sharpness: anti-aliasing threshold for cell to be set.
      num_dim: number of dimensions of box (only 2 for now).

    Returns:
      np.ndarray with total dimensions num_dim with each dimension having
      size `physical_dim / grid_size`.
    """

    array_size = ceil(physical_dim / grid_size)
    count = np.random.randint(1, max_count + 1)
    box = np.zeros([array_size] * num_dim)

    for _ in range(count):
        origin = np.random.randint(0, array_size, num_dim)
        grad = tan(random.uniform(-pi / 2, pi / 2))

        x_left, y_left, x_right, y_right = line_2d_endpoints(array_size, origin, grad)

        rr, cc, val = line_aa(x_left, y_left, x_right, y_right)
        box[rr, cc] += val * 255

    box = (box > sharpness).astype(int)

    return box

def make_dataset(
    type: str,
    count: int,
    shard_size: int = 1000,
    file_prefix: str,
    directory: str,
    *args,
):
    """Writes data of a given type in a directory as .npy shards.

    Each shard file contains a np.ndarray of np.ndarray. Each np.ndarray element
    of the parent np.ndarray is a dataset of randomly generated shapes of `type`
    where set cells are '1' and empty cells are '0'.

    Args:
      type: shape to generate in data.
      count: total number of datasets.
      shard_size: number of datasets in a shard.
      file_prefix: prefix to be appended before shard index.
      directory: location to save.
      args: parameters to pass into the corresponding type_fn (e.g. circle_fn
            if type == 'circle' and line_fn if type == 'line').
    """

    shape_fn = None
    if (type == 'circle'):
        shape_fn = circle_fn
    elif (type == 'line'):
        shape_fn = line_fn
    else:
        raise ValueError("type must be in %s" % str(VALID_TYPES))

    num_files = ceil(float(count) / shard_size)
    for i in range(num_files):
        output = [shape_fn(*args) for _ in range(shard_size)]
        file_path = "%s/%s_%d_of_%d.npy" % (directory, file_prefix, i+1, num_files)
        np.save(file_path, output)
