import os
import unittest

from parameterized import parameterized

from taiyaki import helpers
from util import MODEL_FILES, MODELS_DIR


class TestLoadModel(unittest.TestCase):

    @parameterized.expand(MODEL_FILES)
    def test_model_file_exists(self, filename):
        self.assertTrue(os.path.exists(os.path.join(MODELS_DIR, filename)))

    @parameterized.expand(MODEL_FILES)
    def test_load_model_from_file_with_no_metadata(self, filename):
        helpers.load_model(os.path.join(MODELS_DIR, filename))

    @parameterized.expand(MODEL_FILES)
    def test_load_model_from_file_with_metadata(self, filename):
        metadata = {'reverse': False, 'standardize': False}
        filepath = os.path.join(MODELS_DIR, filename)
        net = helpers.load_model(filepath, model_metadata=metadata)
        self.assertEqual(metadata['reverse'], net.metadata['reverse'])
        self.assertEqual(metadata['standardize'], net.metadata['standardize'])

    @parameterized.expand(MODEL_FILES)
    def test_load_model_from_file_with_wrong_metadata(self, filename):
        metadata = {'reverse': True, 'standardize': False}
        filepath = os.path.join(MODELS_DIR, filename)
        with self.assertWarns(RuntimeWarning):
            net = helpers.load_model(filepath, model_metadata=metadata)
        self.assertEqual(metadata['reverse'], net.metadata['reverse'])
        self.assertEqual(metadata['standardize'], net.metadata['standardize'])
