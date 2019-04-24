"""Utilities for interacting with files on google cloud."""

import tempfile
import os
import pathlib

import matplotlib.pyplot as plt

from google.cloud import storage

def save_fig(
    fig: plt.Figure,
    bucket,
    file_path_in_bucket: str,
):
  """Save matplotlib figure to google cloud directory."""
  with tempfile.TemporaryDirectory() as temp_dir:
    temp_fig_dir = os.path.join(temp_dir, "temp.png")
    fig.savefig(temp_fig_dir)

    print("file name ", file_path_in_bucket)

    # Init GCS client and upload file.
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    blob = bucket.blob(file_path_in_bucket)
    blob.upload_from_filename(filename=temp_fig_dir)


def parse_directory(path):
  """Parses directory and returns bucket and file path if google cloud dir.

  Returns:
    Path: path to file either locally or in google storage bucket.
    bucket: either the bucket on google cloud or None if the path is local.
  """
  p = pathlib.Path(path)

  if p.parts[0] == "gs:":
    return str(pathlib.Path(* p.parts[2:-1], p.stem)), str(p.parts[1])
  else:
    return str(p), None


def maybe_save_cloud(fig, path):
  """Saves either to local directory or google cloud."""
  file_path, google_bucket = parse_directory(path)
  if google_bucket is not None:
    file_path = file_path + ".png"
    save_fig(fig, google_bucket, file_path)
  else:
    fig.savefig(file_path)