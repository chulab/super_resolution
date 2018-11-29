"""Functions to generate datsets and save."""

import sys

import numpy as np
import six
import tensorflow as tf





def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))