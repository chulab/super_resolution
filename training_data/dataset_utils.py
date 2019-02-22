"""Functions for preparing and loading datasets."""

import numpy as np
import tensorflow as tf

_TRAIN = "TRAIN"
_EVAL = "EVAL"
_PREDICT = "PREDICT"


def array_input_fn(array: np.ndarray, mode: str, batch_size: int):
  """Function to prepare array as `tf.dataset.Dataset`."""
  if len(array.shape) != 3:
    raise ValueError("`array` must have shape [n_samples, height, width].")
  _assert_valid_mode(mode)

  dataset = tf.data.Dataset.from_tensor_slices(array)

  if mode == _TRAIN:
    dataset = dataset.shuffle(1000).repeat()

  dataset = dataset.batch(batch_size)

  return dataset

def _assert_valid_mode(mode:str):
  """Asserts `mode` is valid."""
  if not mode in [_TRAIN, _EVAL, _PREDICT]:
    raise ValueError("Invalid mode.")