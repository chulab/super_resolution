"""Saves single slice of numpy array as png image.

Saves image in same location as the numpy array.

Example usage:
  # python save_demo_image.py \
  # -f $FILE_PATH$ \
  # -gd .1e-3 .1e-3 \

"""

import argparse
import os

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar import scalebar

def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-f',
    '--file',
    dest='numpy_file',
    help='Path to numpy file.',
    type=str,
    required=True,
    )


  parser.add_argument(
    '-gd',
    '--grid_dimension',
    dest='grid_dimension',
    help='Grid dimensions.',
    type=float,
    nargs=2,
    required=False,
    )

  return parser.parse_args()

def main():
  args = parse_args()
  file_directory = os.path.dirname(args.numpy_file)

  arr = np.load(args.numpy_file)

  print("Array shape {}".format(arr.shape))

  image_filename = os.path.join(file_directory, "test_image.png")

  fig = plt.figure()
  im = plt.imshow(arr[0], interpolation=None)
  fig.colorbar(im)
  fig.suptitle("Sample distribution with {} pixels.".format(str(arr.shape[1:])),
               fontsize=20,)

  if args.grid_dimension:
    sb = scalebar.ScaleBar(args.grid_dimension[0])
    plt.gca().add_artist(sb)

  plt.savefig(image_filename)

if __name__ == "__main__":
  main()