"""Tests for `save_utils`"""

import unittest

from cloud import save_utils

class testSaveUtils(unittest.TestCase):

  def testParseDirectory(self):
    normal_dir = "a/test/dir"

    split_normal = save_utils.parse_directory(normal_dir)

    self.assertEqual(normal_dir, split_normal[0])
    self.assertEqual(None, split_normal[1])

  def testParseDirectoryGoogle(self):
    google_cloud_dir = "gs://test_bucket/folders/in/storage"

    split_google = save_utils.parse_directory(google_cloud_dir)

    self.assertEqual("folders/in/storage", split_google[0])
    self.assertEqual("test_bucket", split_google[1])