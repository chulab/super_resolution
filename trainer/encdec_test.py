import pytest
import tensorflow as tf
from trainer.encdec_model import ImagesEncoder, LSTMDecoder, EncoderDecoder, get_sinusoidal_embeddings

class EncoderTest(tf.test.TestCase):

    def setUp(self):
        self.frequencies = tf.random.uniform([5, 10])
        self.angles = tf.random.uniform([5, 10])
        self.images = tf.random.uniform([5, 25, 25, 10])
        hparams = EncoderDecoder.get_tiny_hparams()
        hparams.enc_height = 5
        hparams.enc_width = 5
        hparams.enc_channels = 3
        self.hparams = hparams
        self.features = {"frequencies": self.frequencies,
                "angles": self.angles,
                "images": self.images,
                }
        self.encoder = ImagesEncoder(self.hparams, 'CPU')

    def testSinusoidalEmbedding(self):
        emb = get_sinusoidal_embeddings(self.frequencies, 20)

        with self.session():
          tf.global_variables_initializer().run()
          emb.eval()

        self.assertEqual(emb.shape, (5, 10, 20))

    def testFreqAngleEmbedding(self):
        inputs = tf.random.uniform([5, 10, 3])
        emb_dim = 6
        emb = ImagesEncoder.add_freq_angle_embeddings(inputs, self.frequencies,
            self.angles, emb_dim)

        with self.session():
          tf.global_variables_initializer().run()
          emb.eval()

        self.assertEqual(emb.shape, (5, 10, 3 + 2 * emb_dim))

    def testEncode(self):
        features = self.features
        output = self.encoder(features)
        self.assertEqual(output[0].shape, (5, self.hparams.enc_height
            * self.hparams.enc_width * self.hparams.enc_channels))

class DecoderTest(tf.test.TestCase):

    def setUp(self):
        self.frequencies = tf.random.uniform([5, 10])
        self.angles = tf.random.uniform([5, 10])
        self.images = tf.random.uniform([5, 25, 25, 10])
        hparams = EncoderDecoder.get_tiny_hparams()
        hparams.enc_height = 5
        hparams.enc_width = 5
        hparams.enc_channels = 3
        self.hparams = hparams
        self.features = {"frequencies": self.frequencies,
                "angles": self.angles,
                "images": self.images,
                }
        hidden_size = hparams.enc_height * hparams.enc_width * hparams.enc_channels
        self.encoder_output = (tf.random.uniform([5, hidden_size]),
            tf.random.uniform([5, hidden_size]), tf.random.uniform([5, hidden_size]))
        self.decoder = LSTMDecoder(self.hparams, 'CPU')

    def testDecode(self):
        decoder_output = self.decoder(self.encoder_output, self.features)
        self.assertEqual(decoder_output.shape, self.features["images"].shape)
