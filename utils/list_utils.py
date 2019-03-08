"""Utility functions to operate on `List`."""

import re

from typing import List

def _maybe_int(s):
  """If string represents an integer returns the parsed `int`."""
  try:
    return int(s)
  except ValueError:
    return s


def alphanum_key(s: str) -> List:
  """Turn a string into a list of string and number chunks.

  Explicitly:
    "z23a" -> ["z", 23, "a"]

  Args:
    s: string to be split into number and string chunks.

  Returns:
    List of chunks.
  """
  return [_maybe_int(c) for c in re.split('([0-9]+)', s)]


def human_sort(l):
  """ Sort the given list in the way that humans expect."""
  l.sort(key=alphanum_key)