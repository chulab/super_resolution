"""Utilities for interacting with files on google cloud."""

import tempfile
import os
import matplotlib.pyplot as plt
from tensorflow.python.lib.io import file_io
from google.cloud import storage
import binascii
import collections
import datetime
import hashlib
import sys
from six.moves.urllib.parse import quote
from google.oauth2 import service_account
from oauth2client.client import GoogleCredentials
from googleapiclient.discovery import build
import pathlib
import string
import random
import json

def random_string(length=10):
  """Generate a random string of letters and digits"""
  password_characters = string.ascii_letters + string.digits
  return ''.join(random.choice(password_characters) for i in range(length))


def parse_cloud_directory(path):
  """Parses directory and returns bucket and file path if google cloud dir.

  Returns:
    Path: path to file either locally or in google storage bucket.
    bucket: either the bucket on google cloud or None if the path is local.
  """
  p = pathlib.Path(path)
  if p.parts[0] == "gs:":
    return str(pathlib.Path(* p.parts[2:])), str(p.parts[1])
  else:
    return str(p), None


class CredentialsHelper():
  """Helper for Google Credentials."""
  def __init__(self, service_account_path, cloud_train=False):
      self.cloud_train = cloud_train
      if cloud_train:
        service_account_info = json.load(file_io.FileIO(service_account_path,
          'r'))
        self.creds = service_account.Credentials.from_service_account_info(
          service_account_info)
      else:
        self.creds = service_account.Credentials.from_service_account_file(
          self.service_account_path)


  def get_credentials(self):
      return self.creds


  def create_service_account_path(self, service_account_path, cloud_train=False):
    if cloud_train:
      storage_client = storage.Client()
      file_path, google_bucket = parse_cloud_directory(service_account_path)
      bucket = storage_client.get_bucket(google_bucket)
      blob = bucket.blob(file_path)
      dest_path = random_string() + ".json"
      blob.download_to_filename(dest_path)
      return dest_path
    else:
      return service_account_path


class StorageHelper():
  """Helper for Google Storage."""
  def __init__(self, creds):
    self.creds = creds
    self.client = storage.Client(credentials=creds)


  def find_blob(self, full_bucket_path):
    file_path, google_bucket = parse_cloud_directory(full_bucket_path)
    bucket = self.client.get_bucket(google_bucket)
    blob = bucket.blob(file_path)
    return blob


  def upload_file(self, source_path, full_bucket_path):
    blob = self.find_blob(full_bucket_path)
    blob.upload_from_filename(filename=source_path)


  def delete_file(self, full_bucket_path):
    blob = self.find_blob(full_bucket_path)
    blob.delete()


  def list_blobs_with_prefix(self, full_bucket_path, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        /a/1.txt
        /a/b/2.txt

    If you just specify prefix = '/a', you'll get back:

        /a/1.txt
        /a/b/2.txt

    However, if you specify prefix='/a' and delimiter='/', you'll get back:

        /a/1.txt

    """

    prefix, bucket_name = parse_cloud_directory(full_bucket_path)
    bucket = self.client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    full_paths = ["gs://" + os.path.join(bucket_name, blob.name) for blob in blobs]

    return full_paths


  def save_fig(
      self,
      fig: plt.Figure,
      full_bucket_path: str,
  ):
    """Save matplotlib figure to google cloud directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_fig_dir = os.path.join(temp_dir, "temp.png")
      fig.savefig(temp_fig_dir)
      self.upload_file(temp_fig_dir, full_bucket_path)


  def upload_maybe_fig(
      self,
      fig_or_path,
      full_bucket_path: str,
  ):
    if isinstance(fig_or_path, plt.Figure):
        self.save_fig(fig_or_path, full_bucket_path)
    else:
        self.upload_file(fig_or_path, full_bucket_path)


  def maybe_save_cloud(self, fig, path):
    """Saves either to local directory or google cloud."""
    file_path, google_bucket = parse_cloud_directory(path)
    if google_bucket is not None:
      self.save_fig(fig, google_bucket, file_path)
    else:
      fig.savefig(file_path)


  def generate_signed_url(self, full_bucket_path, expiration, http_method='GET'
                      , query_parameters=None, headers=None):
    """
    Generates temporary public url for sharing.

    Arguments:
        full_bucket_path: full path in bucket.
        expiration: time till expiration in seconds (max 604800).
    """

    if expiration > 604800:
        print('Expiration Time can\'t be longer than 604800 seconds (7 days).')
        sys.exit(1)

    object_name, bucket_name = parse_cloud_directory(full_bucket_path)
    escaped_object_name = quote(object_name, safe='')
    canonical_uri = '/{}/{}'.format(bucket_name, escaped_object_name)

    datetime_now = datetime.datetime.utcnow()
    request_timestamp = datetime_now.strftime('%Y%m%dT%H%M%SZ')
    datestamp = datetime_now.strftime('%Y%m%d')

    client_email = self.creds.service_account_email
    credential_scope = '{}/auto/storage/goog4_request'.format(datestamp)
    credential = '{}/{}'.format(client_email, credential_scope)

    if headers is None:
        headers = dict()
    headers['host'] = 'storage.googleapis.com'

    canonical_headers = ''
    ordered_headers = collections.OrderedDict(sorted(headers.items()))
    for k, v in ordered_headers.items():
        lower_k = str(k).lower()
        strip_v = str(v).lower()
        canonical_headers += '{}:{}\n'.format(lower_k, strip_v)

    signed_headers = ''
    for k, _ in ordered_headers.items():
        lower_k = str(k).lower()
        signed_headers += '{};'.format(lower_k)
    signed_headers = signed_headers[:-1]  # remove trailing ';'

    if query_parameters is None:
        query_parameters = dict()
    query_parameters['X-Goog-Algorithm'] = 'GOOG4-RSA-SHA256'
    query_parameters['X-Goog-Credential'] = credential
    query_parameters['X-Goog-Date'] = request_timestamp
    query_parameters['X-Goog-Expires'] = expiration
    query_parameters['X-Goog-SignedHeaders'] = signed_headers

    canonical_query_string = ''
    ordered_query_parameters = collections.OrderedDict(
        sorted(query_parameters.items()))
    for k, v in ordered_query_parameters.items():
        encoded_k = quote(str(k), safe='')
        encoded_v = quote(str(v), safe='')
        canonical_query_string += '{}={}&'.format(encoded_k, encoded_v)
    canonical_query_string = canonical_query_string[:-1]  # remove trailing ';'

    canonical_request = '\n'.join([http_method,
                                   canonical_uri,
                                   canonical_query_string,
                                   canonical_headers,
                                   signed_headers,
                                   'UNSIGNED-PAYLOAD'])

    canonical_request_hash = hashlib.sha256(
        canonical_request.encode()).hexdigest()

    string_to_sign = '\n'.join(['GOOG4-RSA-SHA256',
                                request_timestamp,
                                credential_scope,
                                canonical_request_hash])

    signature = binascii.hexlify(
        self.creds.signer.sign(string_to_sign)
    ).decode()

    host_name = 'https://storage.googleapis.com'
    signed_url = '{}{}?{}&X-Goog-Signature={}'.format(host_name, canonical_uri,
                                                      canonical_query_string,
                                                      signature)
    return signed_url


class SlidesHelper():
  """
  Helper for Google Slides. All member functions append requests for a
  batchUpdate request to ensure concurrency. It is the responsibility of the
  caller to call execute() to send the batchUpdate.
  """
  def __init__(self, pres_id, creds):
      self.service= build('slides', 'v1', credentials=creds)
      self.requests = []
      self.pres = self.service.presentations().get(
          presentationId=pres_id).execute()
      self.pres_id = pres_id
      self.add = 0


  def find_element_by_text(self, slide, text, shapeType=None):
    for element in slide['pageElements']:
      if shapeType is None or element['shape']['shapeType'] == shapeType:
        for t_elem in element['shape']['text']['textElements']:
          if 'textRun' in t_elem and text in t_elem['textRun']['content']:
              return element
    return None


  def duplicate_slide(self, slide_no):
      """
      Creates a duplicate of a slide, with the new slide being below the
      original.
      """

      page_id = self.pres['slides'][slide_no]['objectId']
      note_id = self.pres['slides'][slide_no]['slideProperties']['notesPage'] \
          ['notesProperties']['speakerNotesObjectId']
      new_id = random_string()
      new_note_id = random_string()
      req = {
        "duplicateObject": {
          "objectId": page_id,
          "objectIds": {
            page_id : new_id,
            note_id: new_note_id,
          }
        }
      }
      self.requests.append(req)
      self.add += 1

      return self.pres['slides'][slide_no]['objectId'], note_id, new_id, new_note_id


  def move_to_back(self, slide_id):
      req = {
              "updateSlidesPosition": {
                  "slideObjectIds": [
                  slide_id
                  ],
                  "insertionIndex": len(self.pres['slides']) + self.add
              }
          }
      self.requests.append(req)

  def replace_text(self, slide_id, search_string, texts,
      labels=None, separator=':'):
      """
      Replaces objects containing search_string with new text given in texts.
      If labels are given, each label is placed before each text in texts
      and separated by the separator.

      Arguments:
          slide_id: slide json object.
          search_string: string to search for.
          texts: array of texts to be used.
          labels: title of each text.
          separator: character between labels and texts.
      """

      if labels:
          new_text = ''.join([label + separator + ' ' + text + '\n' for
              text, label in zip(texts,labels)])
      else:
          new_text = ''.join([text + '\n' for text in texts])
      req = {
        "replaceAllText": {
          "containsText": {
            "text": search_string,
            "matchCase": False
          },
          "replaceText": new_text,
          "pageObjectIds": [slide_id]
        }
      }
      self.requests.append(req)


  def replace_image(self, slide_id, search_string, img_url):
      req = {
        'replaceAllShapesWithImage': {
          'imageUrl': img_url,
          'imageReplaceMethod': 'CENTER_CROP',
          'pageObjectIds': [slide_id],
          'containsText': {
            "text": search_string,
            "matchCase": False
          }
        }
      }
      self.requests.append(req)


  def replace_texts_and_images(self, slide_id, texts_array, img_url_array,
          labels_array = None, separator_array = None):

      if labels_array is None:
          labels_array = [None for _ in range(len(texts_array))]
      if separator_array is None:
          separator_array = [':' for _ in range(len(texts_array))]

      for i, (texts, labels, separator) in enumerate(zip(texts_array,
          labels_array, separator_array)):
          self.replace_text(slide_id, 'text{}'.format(i+1), texts, labels, separator)

      for i, img_url in enumerate(img_url_array):
          self.replace_image(slide_id, 'image{}'.format(i+1), img_url)


  def add_comment(self, note_id, comment):
      self.requests.append({
          'insertText': {
              'objectId': note_id,
              'insertionIndex': 0,
              'text': comment
          }
      })


  def refresh(self):
      self.pres = self.service.presentations().get(
      presentationId=self.pres_id).execute()
      self.add = 0


  def execute(self):
      self.service.presentations().batchUpdate(body={'requests': self.requests},
          presentationId=self.pres_id).execute()
      self.requests = []
      self.refresh()


class SlideTemplater():
  """
  Highest level API to create a Google Slide from a summary. See examples of
  summaries in analysis/summaries.py. It is the responsibility of the
  caller to call execute() to properly batch requests.
  """
  def __init__(self, cloud_train, service_account_path, pres_id):
      self.credentials = CredentialsHelper(service_account_path, cloud_train)
      creds = self.credentials.get_credentials()
      self.slides = SlidesHelper(pres_id, creds)
      self.storage = StorageHelper(creds)
      self.cloud_train = cloud_train
      self.paths = []
      self.template_id = None


  def fill_template_from_cloud(self, texts_array, full_bucket_paths,
          labels_array = None, separator_array = None,
          template_no = -1, comment = None, temp_storage = False):
      """
      Fills texts, images and comment according to a template Google slide.
      Example raw and filled templates are given in analysis/example_template
      and analysis/example_template_filled.png.

      Replacing text:
        Searches template slide for textbox with placeholder `text{i}` and
        replaces placeholder with
          `{i}th elem of labels_array + {i}th elem of separator_array +
           {i}th elem of texts_array`.

      Replacing image:
        Searches template slide for rectangle with placeholder `image{i}` and
        fills rectangle with image stored in {i}th element of full_bucket_paths.

      Args:
        texts_array: List of text strings.
        full_bucket_paths: List of Google storage paths of image files.
        labels_array: List of labels for texts_array.
        separator_array: List of separators for texts_array.
        template_no: Index of template Google Slide.
        comment: Comment to add to template.
        temp_storage: Bool denoting whether images in full_bucket_paths should
          be deleted after execution.
      """
      img_url_array = []
      for b_path in full_bucket_paths:
          img_url_array.append(self.storage.generate_signed_url(b_path, 10000))

      slide_id, note_id, new_slide_id, new_note_id = self.slides.duplicate_slide(template_no)
      self.slides.replace_texts_and_images(new_slide_id, texts_array, img_url_array
          , labels_array, separator_array)
      if comment:
          self.slides.add_comment(new_note_id, comment)

      self.template_id = slide_id
      if temp_storage:
        self.paths += full_bucket_paths


  def upload_and_fill_template(self, texts_array, fig_or_path_array,
          full_bucket_paths, labels_array = None, separator_array = None,
          template_no = -1, comment = None, temp_storage = False):
      """
      Uploads figures and fills template.

      Args:
        fig_or_path_array: List of plt.Figures or local filepaths of images.
        full_bucket_paths: List of Google Storage paths to save images to.

      Refer to `fill_template_from_cloud` for other arguments.
      """
      for fig_or_path, b_path in zip(fig_or_path_array, full_bucket_paths):
          self.storage.upload_maybe_fig(fig_or_path, b_path)

      self.fill_template_from_cloud(texts_array, full_bucket_paths,
              labels_array, separator_array, template_no, comment, temp_storage)

  def execute(self):
    self.slides.execute()
    if self.template_id is not None:
      self.slides.move_to_back(self.template_id)
      self.slides.execute()
    self.template_id = None
    for b_path in self.paths:
        self.storage.delete_file(b_path)
