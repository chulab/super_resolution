"""Utils for logging"""

import os

import logging
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def set_up_logging(log_dir: str=''):
  """Sets up logging."""

  # Check for environmental variable.
  file_location = os.getenv('JOB_DIRECTORY', log_dir)

  print("Logging file writing to {}".format(file_location), flush=True)

  logging.basicConfig(
    filename=os.path.join(file_location, 'training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(process)d - %(message)s'
  )

  logging.debug("Initialize debug.")
