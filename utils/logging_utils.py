"""Utils for logging"""

import os

import logging
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def set_up_logging():
  """Sets up logging."""

  # Check for environmental variable.
  file_location = os.getenv('JOB_DIRECTORY', '.')

  print("Logging file writing to {}".format(file_location), flush=True)

  logging.basicConfig(
    filename=os.path.join(file_location, 'training.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(process)d - %(message)s'
  )

  logging.debug("Initialize debug.")

########################### GOOGLE SLIDES ################################

# def slides_initialize(slides_id, cred_dir, cloud_mode):
#     SCOPES = ['https://www.googleapis.com/auth/presentations']
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     token_path = os.path.join(cred_dir, 'slides_token.pickle')
#     cred_path = os.path.join(cred_dir, 'slides_credentials.json')
#
#     if cloud_mode:
#         with file_io.FileIO(token_path, mode='rb') as token:
#             creds = pickle.load(token)
#     else:
#         if os.path.exists(token_path):
#             with open(token_path, 'rb') as token:
#                 creds = pickle.load(token)
#         # If there are no (valid) credentials available, let the user log in.
#         if not creds or not creds.valid:
#             if creds and creds.expired and creds.refresh_token:
#                 creds.refresh(Request())
#             else:
#                 flow = InstalledAppFlow.from_client_secrets_file(
#                     cred_path, SCOPES)
#                 creds = flow.run_local_server()
#             # Save the credentials for the next run
#             with open(token_path, 'wb') as token:
#                 pickle.dump(creds, token)
#
#     DRIVE = build('drive', 'v3', credentials=creds)
#     SLIDES = build('slides', 'v1', credentials=creds)
#
#     # Call the Slides API
#     # presentation = service.presentations().get(
#     #     presentationId=PRESENTATION_ID).execute()
#     # slides = presentation.get('slides')
#     return DRIVE, SLIDES
#
# def execute(slides_id, cred_dir, cloud_mode):
#     DRIVE, SLIDES = slides_initialize(slides_id, cred_dir, cloud_mode)
#     body = []
#     body = add_image(body, 'https://drive.google.com/uc?id=1GaPjlSqYsC1CjqRooManLpPMB9rTSvck', 3000, 3000, 2000, 5000)
#     SLIDES.presentations().batchUpdate(body={'requests': body},
#         presentationId=slides_id).execute()
#
# def add_image(body, source_url, height, width, x, y):
#     request = {
#     'createImage':{
#             'elementProperties':{
#                 'size':{
#                     'height': {'magnitude': height, 'unit': 'EMU'},
#                     'width': {'magnitude': width, 'unit': 'EMU'}
#                 },
#                 'transform': {
#                     'unit': 'EMU', 'translateX': x, 'translateY': y
#                 }
#             },
#             "url": source_url
#         }
#     }
#     body.append(request)
#     return body
#
# execute('187e6QMPZApaDQoqC5v_BqQ4Y0mZUsUD1VkviSu_hCnY', '', False)
