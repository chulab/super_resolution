def draw_blob(
    array_size: int,
    origin,
    avg_radius: int,
    num_dim: int,
    step_size: int = 1,
):
    if step_size % 2 != 1:
        raise ValueError("step_size must be odd")

    box = np.zeros([array_size] * num_dim)
    p = 0.8
    neighbor_span = step_size / 2

    # equivalent to [0:array_size, 0:array_size, ...] repeated num_dim times
    args = [slice(i,j) for i,j in [(-1 * neighbor_span, neighbor_span)] * num_dim]
    neighbors = [[i, j, k] for i in range(args)]

    queue = []

    queue.append(origin)
    while (queue):
        point = queue.pop(0)
        box[origin] = 1
        for n in neighbors:
            box[origin + n] = 1

def make_dataset(
    type: str,
    count: int,
    file_prefix: str,
    directory: str,
    shard_size: int = 1000,
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
        raise ValueError("type must be in %s." % str(VALID_TYPES))

    num_files = ceil(float(count) / shard_size)
    for i in range(num_files):
        output = [shape_fn(*args) for _ in range(shard_size)]
        file_path = "%s/%s_%d_of_%d.npy" % (directory, file_prefix, i+1, num_files)
        np.save(file_path, output)

def _blob(
    coordinates: np.ndarray,
    origin: List[float],
    radius: float,
    grid_dimensions: List[float],
    step_size: int = 1,
):
    if step_size % 2 != 1:
        raise ValueError("step_size must be odd")

    box = np.zeros(coordinates.shape[:-1])

    scaled_radius = radius / (radius + 1)
    #walk_probs = [scaled_radius / length for length in grid_dimensions]

    neighbor_span = int(step_size / 2)

    # equivalent to [0:array_size, 0:array_size, ...] repeated num_dim times
    #args = [slice(i,j) for i,j in [(-1 * neighbor_span, neighbor_span)] * box.ndim]
    #print(args)
    #neighbors = [[i, j, k] for i in range(args)]
    neighbors = np.stack(
      response_functions.coordinate_grid([2 * neighbor_span] * box.ndim
      , [1] * box.ndim, True)
      , -1
    ).astype(int)

    possible_nexts = np.stack(
      response_functions.coordinate_grid([2 * step_size] * box.ndim
      , [step_size] * box.ndim, True)
      , -1
    ).astype(int)

    # possible_nexts_physical = np.stack(
    #   response_functions.coordinate_grid(np.asarray(grid_dimensions) * 2 * step_size
    #   , np.asarray(grid_dimensions) * step_size, True)
    #   , -1
    # ).astype(int)

    norms = np.sqrt(np.sum((possible_nexts * grid_dimensions) ** 2, -1))
    walk_probs = np.where(norms != 0, scaled_radius/norms/3, 0)

    grid_origin = find_closest(coordinates, origin)
    queue = []
    queue.append(grid_origin)

    while (len(queue) > 0):
        point = queue.pop(0)
        for x in np.nditer(np.moveaxis(point + neighbors, -1 ,0), flags=['external_loop'], order='F'):
            if within_bounds(box, x):
                box[tuple(x)] = 1
        #for p, next in zip(walk_probs, point + possible_nexts):
        # for (p, next) in np.nditer([walk_probs, point + possible_nexts], order='F'):
        nexts = point + possible_nexts
        it = np.nditer(walk_probs, flags=['multi_index'], order='F')
        while not it.finished:
            p = it[0]
            next = nexts[it.multi_index]
            coin = np.random.uniform(0, 1)
            if (coin <= p) and within_bounds(box, next) and box[tuple(next)] == 0:
                queue.append(next)
            it.iternext()

    print(box)

# coordinates = np.stack(
#   response_functions.coordinate_grid([20., 20.], [1., 1.], False),
#   -1
# )
# _blob(coordinates, [10., 10.], 2., [1., 1.], 1)

def find_closest(coordinates: np.ndarray, vector):
  """Returns index in coordinates closest to vector"""
  distance_squared = np.sum((coordinates - vector) ** 2, -1)
  flattened_index = np.argmin(distance_squared, axis=None)
  return np.unravel_index(flattened_index, distance_squared.shape)

def within_bounds(box: np.ndarray, index: np.ndarray):
    result = True
    for i, length in enumerate(box.shape):
        if (index[i] < 0) or (index[i] >= length):
            result = False
            break
    return result
