import argparse
import logging
import copy
import math
import tensorflow as tf
from typing import Tuple
from trainer import blocks
from preprocessing import preprocess
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.models import transformer
from simulation import create_observation_spec
from tensor2tensor.layers import common_layers
import tensorflow.keras.backend as K

def get_sinusoidal_embeddings(features, channels, min_timescale=1.0,
        max_timescale=1.0e4):

    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(features, -1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
    signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [tf.shape(features)[0], features.shape[1], channels])
    return signal

def add_freq_angle_embeddings(inputs, frequencies, angles,
            emb_dim, min_timescale=1.0, max_timescale=1.0e4):
    """Adds sinusoidal frequency and angle embeddings, each of dimension
    emb_dim, to inputs

    Args:
        inputs: `tf.Tensor` with shape [batch, num_images, hidden_size]
        frequencies: `tf.Tensor` with shape [batch, num_images]
        angles: `tf.Tensor` with shape [batch, num_images]
        emb_dim: int

    Returns:
      `tf.Tensor` with shape [batch, num_images, hidden_size + 2 * emb_dim]
    """
    freq_emb = get_sinusoidal_embeddings(frequencies, emb_dim, min_timescale, max_timescale)
    angle_emb = get_sinusoidal_embeddings(angles, emb_dim, min_timescale, max_timescale)
    embedded = tf.concat([inputs, freq_emb, angle_emb], axis=-1)

    return embedded

class ImagesEncoder(tf.keras.Model):

    def __init__(self, hparams, device='GPU'):
        super(ImagesEncoder, self).__init__()
        self.hparams = hparams
        feed_forward = tf.keras.models.Sequential()
        for _ in range(hparams.enc_ff_layers):
            feed_forward.add(tf.keras.layers.Dense(hparams.enc_height*hparams.enc_width*hparams.enc_channels))
        feed_forward.add(tf.keras.layers.Dense(hparams.enc_height*hparams.enc_width*hparams.enc_channels, activation='tanh'))
        self.feed_forward = feed_forward
        if device == 'GPU':
            lstm = tf.keras.layers.CuDNNLSTM(hparams.enc_height * hparams.enc_width * hparams.enc_channels, return_state=True)
        else:
            lstm = tf.keras.layers.LSTM(hparams.enc_height * hparams.enc_width * hparams.enc_channels, return_state=True)
        self.lstm = lstm
        # self.lstm = tf.keras.layers.Bidirectional(lstm)
        self.convs = []
        for _ in range(hparams.enc_conv_blocks):
            self.convs.append(tf.keras.layers.Conv2D(
              filters=hparams.enc_conv_filters,
              kernel_size=hparams.enc_conv_block_kernel_size,
              padding="same",
              use_bias=True,
              activation=tf.nn.leaky_relu
            ))
        self.postconv = tf.keras.layers.SeparableConv2D(
          filters=hparams.mid_channels,
          kernel_size=[3, 3],
          dilation_rate=1,
          padding="same",
          activation=tf.nn.leaky_relu
        )
        self.downsamples = []
        for _ in range(hparams.downsample_blocks):
            self.downsamples.append(tf.keras.layers.SeparableConv2D(
              filters=hparams.mid_channels,
              kernel_size=[3, 3],
              dilation_rate=1,
              padding="same",
              strides=2,
              activation=tf.nn.leaky_relu
            ))


    def call(self, inputs):
        """Encodes inputs of shape [batch, height, width, num_images]"""

        hparams = copy.copy(self.hparams)

        storage = []

        for image in tf.unstack(inputs["images"], axis=-1):
            output = tf.expand_dims(image, axis=-1)
            for conv_layer in self.convs:
                output = conv_layer(output)
                output = tf.keras.layers.BatchNormalization().apply(output)
            # print("Conv shape", output.shape)
            output = self.postconv(output)
            # print("Postprocess shape", output.shape)
            for downsample_layer in self.downsamples:
                output = downsample_layer.apply(output)
                output = tf.keras.layers.BatchNormalization().apply(output)
            # print("Downsample shape", output.shape)
            output = tf.reshape(output, [tf.shape(output)[0], 1, output.shape[1]*output.shape[2]*output.shape[3]])
            # print("Final shape", output.shape)
            storage.append(output)

        cnn_output = tf.concat(storage, axis=1)
        # print("Stacked shape", cnn_output.shape)

        frequencies = inputs["frequencies"]
        angles = inputs["angles"]
        embedded = add_freq_angle_embeddings(cnn_output, frequencies, angles, hparams.enc_emb_dim)

        # reshaped = tf.reshape(embedded, [tf.shape(embedded)[0]*embedded.shape[1], embedded.shape[2]])
        # ff_output = self.feed_forward(reshaped)
        # print(embedded.shape)
        # print(reshaped.shape)
        # print(ff_output.shape)
        # ff_output = tf.reshape(ff_output, [tf.shape(embedded)[0], tf.shape(embedded)[1], -1])

        ff_output = self.feed_forward(embedded)
        # ff_storage = []
        # for input in tf.unstack(embedded, axis=1):
        #     ff_out = self.feed_forward(input)
        #     ff_out = tf.expand_dims(ff_out, axis=1)
        #     ff_storage.append(ff_out)
        # ff_output = tf.concat(ff_storage, axis=1)

        encoded, state_h, state_c = self.lstm(ff_output)
        # encoded, state_h1, state_c1, state_h2, state_c2 = self.lstm(embedded)
        # return encoded, state_h1, state_c1, state_h2, state_c2
        return encoded, state_h, state_c

class LSTMDecoder(tf.keras.Model):

    def __init__(self, hparams, device='GPU'):
        super(LSTMDecoder, self).__init__()
        self.hparams = hparams
        self.device = device
        # cells = [tf.nn.rnn_cell.LSTMCell(hparams.enc_height * hparams.enc_width * hparams.enc_channels, state_is_tuple=True) for _ in range(hparams.max_input_length)]
        # self.lstm = tf.keras.layers.RNN(cells)
        # self.lstm = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        # cells = [tf.keras.layers.LSTMCell(hparams.enc_height * hparams.enc_width * hparams.enc_channels)] * hparams.max_input_length
        # self.lstm = tf.keras.layers.RNN(cells)
        if device == 'GPU':
            lstm = tf.keras.layers.CuDNNLSTM(hparams.enc_height * hparams.enc_width * hparams.enc_channels, return_sequences=True)
        else:
            lstm = tf.keras.layers.LSTM(hparams.enc_height * hparams.enc_width * hparams.enc_channels, return_sequences=True)
        self.lstm = lstm
        # self.lstm2 = tf.keras.layers.Bidirectional(lstm)
        self.convs = []
        for _ in range(hparams.dec_conv_blocks):
            self.convs.append(tf.keras.layers.Conv2DTranspose(
              filters= hparams.dec_conv_filters,
              kernel_size=hparams.dec_conv_block_kernel_size,
              padding="same",
              use_bias=True,
              activation=tf.nn.leaky_relu
            ))
        self.postconv = tf.keras.layers.SeparableConv2D(
          filters= 1,
          kernel_size=[3, 3],
          dilation_rate=1,
          padding="same",
          activation=tf.nn.leaky_relu
        )


    def call(self, encoder_output, inputs):
        hparams = copy.copy(self.hparams)
        encoded, state_h, state_c = encoder_output
        # encoded, state_h1, state_c1, state_h2, state_c2 = encoder_output
        image_shape = inputs["images"].shape
        freq_emb = get_sinusoidal_embeddings(inputs["frequencies"], hparams.dec_emb_dim)
        angle_emb = get_sinusoidal_embeddings(inputs["angles"], hparams.dec_emb_dim)
        lstm_input = tf.concat([freq_emb, angle_emb], axis=-1)
        if self.device == "GPU":
            # self.lstm.state = tuple([h_state, c_state])
            lstm_output = self.lstm(lstm_input, initial_state=[state_h, state_c])
            # lstm_output = self.lstm(lstm_input, initial_state=[[state_h1, state_c1], [state_h2, state_c2]])
        else:
            # self.lstm.forward_layer.states[0] = state_h1
            # self.lstm.forward_layer.states[1] = state_c1
            # self.lstm.backward_layer.states[0] = state_h2
            # self.lstm.backward_layer.states[1] = state_c2
            # self.lstm.states[1] = c_state
            self.lstm.states[0] = state_h
            self.lstm.states[1] = state_c
            lstm_output = self.lstm(lstm_input)
        storage = []
        for element in tf.unstack(lstm_output, axis=1):
            image = tf.reshape(element, [tf.shape(element)[0], hparams.enc_height, hparams.enc_width, hparams.enc_channels])
            output = tf.image.resize_bilinear(image, image_shape[1:3])
            for conv_layer in self.convs:
                output = conv_layer.apply(output)
                output = tf.keras.layers.BatchNormalization().apply(output)
            output = self.postconv.apply(output)
            storage.append(output)
        decoder_output = tf.concat(storage, axis=-1)
        return decoder_output

class FeedForwardDecoder(tf.keras.Model):
    def __init__(self, hparams):
        super(FeedForwardDecoder, self).__init__()
        self.hparams = hparams
        feed_forward = tf.keras.models.Sequential()
        for _ in range(hparams.dec_ff_layers):
            feed_forward.add(tf.keras.layers.Dense(hparams.enc_height*hparams.enc_width*hparams.enc_channels))
        feed_forward.add(tf.keras.layers.Dense(hparams.enc_height*hparams.enc_width*hparams.enc_channels, activation='tanh'))
        self.feed_forward = feed_forward
        self.convs = []
        for _ in range(hparams.dec_conv_blocks):
            self.convs.append(tf.keras.layers.Conv2DTranspose(
              filters= hparams.dec_conv_filters,
              kernel_size=hparams.dec_conv_block_kernel_size,
              padding="same",
              use_bias=True,
              activation=tf.nn.leaky_relu
            ))
        self.postconv = tf.keras.layers.SeparableConv2D(
          filters= 1,
          kernel_size=[3, 3],
          dilation_rate=1,
          padding="same",
          activation=tf.nn.leaky_relu
        )

    def call(self, encoder_output, inputs):
        hparams = copy.copy(self.hparams)
        encoded, state_h, state_c = encoder_output
        image_shape = inputs["images"].shape
        freq_emb = get_sinusoidal_embeddings(inputs["frequencies"], hparams.dec_emb_dim)
        angle_emb = get_sinusoidal_embeddings(inputs["angles"], hparams.dec_emb_dim)
        angle_freq_emb = tf.concat([freq_emb, angle_emb], axis=-1)
        storage = []
        for element in tf.unstack(angle_freq_emb, axis=1):
            ff_input = tf.concat([encoded, element], axis=-1)
            ff_ouput = self.feed_forward(ff_input)
            image = tf.reshape(ff_ouput, [tf.shape(ff_ouput)[0], hparams.enc_height, hparams.enc_width, hparams.enc_channels])
            output = tf.image.resize_bilinear(image, image_shape[1:3])
            for conv_layer in self.convs:
                output = conv_layer.apply(output)
                output = tf.keras.layers.BatchNormalization().apply(output)
            output = self.postconv.apply(output)
            storage.append(output)
        decoder_output = tf.concat(storage, axis=-1)
        return decoder_output

class EncoderDecoder(tf.keras.Model):

    def __init__(self, hparams, device='GPU'):
        super(EncoderDecoder, self).__init__()
        self.encoder = ImagesEncoder(hparams, device)
        self.decoder = LSTMDecoder(hparams, device)

    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output, inputs)
        return decoder_output

    @classmethod
    def get_tiny_hparams(cls):
        hparams = tf.contrib.training.HParams(
            enc_conv_blocks=2,
            enc_conv_block_kernel_size=5,
            enc_conv_filters=72,
            enc_emb_dim=5,
            enc_height = 10,
            enc_width = 10,
            enc_channels = 3,
            mid_channels = 10,
            downsample_blocks = 1,
            dec_conv_blocks=2,
            dec_conv_filters=72,
            dec_conv_block_kernel_size=5,
            dec_emb_dim=5,
            max_input_length=20,
        )
        return hparams

    @classmethod
    def get_base_hparams(cls):
        hparams = tf.contrib.training.HParams(
            enc_conv_blocks=5,
            enc_conv_block_kernel_size=5,
            enc_conv_filters=72,
            enc_emb_dim=50,
            enc_height = 20,
            enc_width = 20,
            enc_channels = 3,
            mid_channels = 20,
            downsample_blocks = 0,
            dec_conv_blocks=5,
            dec_conv_filters=72,
            dec_conv_block_kernel_size=5,
            dec_emb_dim=30,
            max_input_length=80,
            enc_ff_layers=0,
        )
        return hparams

class EncoderFFDecoder(tf.keras.Model):

    def __init__(self, hparams, device='GPU'):
        super(EncoderFFDecoder, self).__init__()
        self.encoder = ImagesEncoder(hparams, device)
        self.decoder = FeedForwardDecoder(hparams)

    def call(self, inputs):
        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output, inputs)
        return decoder_output

    @classmethod
    def get_tiny_hparams(cls):
        hparams = tf.contrib.training.HParams(
            enc_conv_blocks=2,
            enc_conv_block_kernel_size=5,
            enc_conv_filters=72,
            enc_emb_dim=5,
            enc_height = 10,
            enc_width = 10,
            enc_channels = 3,
            mid_channels = 10,
            downsample_blocks = 1,
            dec_conv_blocks=2,
            dec_conv_filters=72,
            dec_conv_block_kernel_size=5,
            dec_emb_dim=5,
            max_input_length=20,
            enc_ff_layers=1,
            dec_ff_layers=1,
        )
        return hparams

    @classmethod
    def get_base_hparams(cls):
        hparams = tf.contrib.training.HParams(
            enc_conv_blocks=5,
            enc_conv_block_kernel_size=5,
            enc_conv_filters=72,
            enc_emb_dim=50,
            enc_height = 20,
            enc_width = 20,
            enc_channels = 3,
            mid_channels = 20,
            downsample_blocks = 0,
            dec_conv_blocks=5,
            dec_conv_filters=72,
            dec_conv_block_kernel_size=5,
            dec_emb_dim=50,
            max_input_length=80,
            enc_ff_layers=0,
            dec_ff_layers=3,
        )
        return hparams

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  hparams = EncoderDecoder.get_base_hparams()
  hparams.add_hparam("learning_rate", 0.001)
  hparams.add_hparam("observation_spec", None)
  hparams.add_hparam("observation_pool_downsample", 10)
  hparams.add_hparam("distribution_pool_downsample", 10)
  hparams.add_hparam("bit_depth", 2)
  hparams.add_hparam("decay_step", 500)
  hparams.add_hparam("decay_rate", 0.9)
  hparams.add_hparam("device", "GPU")
  hparams.add_hparam("seed", 1)
  hparams.add_hparam("length", 12)
  return hparams

def gpu_preprocess(observations, distributions, mode, params):

  distributions, observations = preprocess.hilbert(hilbert_axis=2)(distributions, observations)
  raw_observations = observations

  distributions = distributions[ ..., tf.newaxis]
  distributions = tf.keras.layers.AveragePooling2D(
    params.distribution_pool_downsample).apply(distributions) * (
      params.distribution_pool_downsample ** 2)
  distributions = distributions[..., 0]

  angles = params.observation_spec.angles
  psfs = params.observation_spec.psf_descriptions

  observations = tf.split(observations, observations.shape[1], 1)

  observation_pooling_layer = tf.keras.layers.AveragePooling2D(
    params.observation_pool_downsample)
  observations = [
    observation_pooling_layer.apply(tf.squeeze(o, axis=1)) for o in observations]

  observations = [
    tf.contrib.image.rotate(tensor, -1 * ang, interpolation='BILINEAR')
    for tensor, ang in zip(observations, angles)
  ]
  observations = tf.keras.layers.Concatenate(axis=-1).apply(observations)
  observations.set_shape([None, distributions.shape[1], distributions.shape[2],
    len(params.observation_spec.angles) * len(params.observation_spec.psf_descriptions)])
  angle_info = [tf.ones_like(raw_observations[:,0,0,0,:]) * ang for ang in angles]
  angle_info = tf.concat(angle_info, -1)
  frequency_info = [tf.fill([tf.shape(observations)[0], len(angles)], psf.frequency) for psf in psfs]
  frequency_info = tf.concat(frequency_info, -1)

  if mode == 'TRAIN':
    random_slice = tf.random_uniform([params.length], minval=0, maxval=observations.shape[-1], dtype=tf.dtypes.int32, seed=params.seed)
  else:
    random_slice = tf.linspace(0., observations.get_shape().as_list()[-1]-1, params.length)
    random_slice = tf.cast(random_slice, dtype=tf.int32)

  random_observations = tf.gather(observations, random_slice, axis=-1)
  random_angles = tf.gather(angle_info, random_slice, axis=-1)
  random_frequencies = tf.gather(frequency_info, random_slice, axis=-1)

  observations = {'images':random_observations, 'angles':random_angles, 'frequencies':random_frequencies}

  return observations, distributions


def model_fn(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: dict containing:
      `images`: a `tf.Tensor` with shape `[batch_size, height, width, channels]`
      `frequencies`: a `tf.Tensor` with shape `[batch_size, channels]`
      `angles`: a `tf.Tensor` with shape `[batch_size, channels]`
    labels: dict containing:
      `distribution`: a `tf.Tensor` with shape `[batch_size, height, width]`
    mode: str. must be one of `tf.estimator.ModeKeys`.
    params: `tf.contrib.training.HParams` object containing hyperparameters for
      model.

  Returns:
    `tf.Estimator.EstimatorSpec` object.
  """
  hooks = []

  observations = features
  logging.info("`observations` tensor recieved in model is "
                "{}".format(observations))
  distributions = labels
  logging.info("`distributions` tensor recieved in model is "
                "{}".format(distributions))

  observations, distributions = gpu_preprocess(observations, distributions, mode, params)

  images = observations["images"]
  image_summaries = []
  for i in range(images.shape[-1]):
      image_summaries.append(tf.summary.image("original_image_{}".format(i+1), images[:, :, :, i][..., tf.newaxis], 1))

  model = EncoderDecoder(params, params.device)
  predictions = model(observations)

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions
    )

  difference = images - predictions

  for i in range(images.shape[-1]):
      image_summaries.append(tf.summary.image("predicted_image_{}".format(i+1), predictions[:, :, :, i][..., tf.newaxis], 1))
      image_summaries.append(tf.summary.image("difference_{}".format(i+1), difference[:, :, :, i][..., tf.newaxis], 1))

  images_summaries = tf.summary.merge(image_summaries)

  eval_summary_hook = tf.train.SummarySaverHook(
    summary_op=images_summaries, save_secs=120)

  # Loss. Compare test_output of nn to original images.
  with tf.variable_scope("loss"):

    loss = tf.losses.mean_squared_error(labels=images, predictions=predictions)

    tf.summary.scalar("mean_square_loss", loss)

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=params.learning_rate,
      global_step=tf.train.get_global_step(),
      decay_steps=params.decay_step,
      decay_rate=params.decay_rate,
      staircase=False,
    )
    tf.summary.scalar("learning_rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(
      loss, global_step=tf.train.get_global_step())

  with tf.variable_scope("metrics"):

    abs_difference = tf.math.abs(difference)
    accuracies = tf.zeros([tf.shape(images)[0]], tf.float32)
    bucket_size = (tf.reduce_max(images) - tf.reduce_min(images)) / 256
    bucket_accuracy = tf.reduce_mean(tf.cast(abs_difference <= bucket_size, tf.float32))
    # for image in tf.split(images, images.shape[0], 0):
    #     bucket_size = (tf.reduce_max(image) - tf.reduce_min(image)) / 256
    #     batch_acc = tf.reduce_mean(
    #       tf.cast(abs_difference <= bucket_size, tf.float32))
    #     accuracies[batch] = batch_acc
    # bucket_accuracy = tf.reduce_mean(accuracies)

    tf.summary.scalar("bucket_accuracy", bucket_accuracy)

    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"bucket_accuracy": bucket_accuracy},
      every_n_iter=50
    )
    hooks.append(accuracy_hook)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predictions,
    eval_metric_ops=None,
    training_hooks=hooks,
    evaluation_hooks=[eval_summary_hook, accuracy_hook],
  )

def input_fns_(
  example_shape: Tuple[int, int],
  observation_spec,
):
  """Input functions for training residual_frequency_first_model."""
  fns =[]

  # Parse.
  fns.append(preprocess.parse())

  # Add shape
  fns.append(preprocess.set_shape(
    distribution_shape=example_shape,
    observation_shape=[len(observation_spec.angles)] + list(example_shape) + [len(observation_spec.psf_descriptions)]))

  # Check for Nan.
  fns.append(preprocess.check_for_nan)

  fns.append(preprocess.swap)

  return fns


def input_fns():
  args = parse_args()

  observation_spec = create_observation_spec.load_observation_spec(
    args.observation_spec_path, args.cloud_train
  )

  return input_fns_(
    example_shape=args.example_shape,
    observation_spec=observation_spec,
  )


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--example_shape',
    type=lambda s: [int(size) for size in s.split(',')],
    required=True,
  )

  parser.add_argument(
    '--observation_spec_path',
    type=str,
    required=True,
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  args, _ = parser.parse_known_args()

  return args
