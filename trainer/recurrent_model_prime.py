import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, ReLU, \
Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization, Input, Lambda, \
Dense, Add, Concatenate, GlobalAveragePooling2D, Reshape, Activation, \
DepthwiseConv2D, SeparableConv2D
from tensorflow.keras import Model
from tensor2tensor.layers import common_image_attention as cia
from typing import Tuple
import argparse
import logging
from simulation import create_observation_spec
from preprocessing import preprocess
from trainer import loss_utils

from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

def make_hparams() -> tf.contrib.training.HParams:
  """Create a HParams object specifying model hyperparameters."""
  return tf.contrib.training.HParams(
    learning_rate=0.001,
    observation_spec=None,
    unet_height=4,
    initial_filters=64,
    unet_type='residual',
    recurrent='None',
    num_split=1,
    dropout=0.,
    observation_pool_downsample=10,
    distribution_pool_downsample=10,
    bit_depth=4,
    decay_step=5000,
    decay_rate=.9,
    bets="-logn",
    rewards="1/sqrtn",
    scale_steps=5000,
    loss_scale=1.04,
    diff_scale="None",
    last_loss_only=False,
    se_blocks=5,
    embedding="conv_attention",
    gating="LSTM",
    model_type="propagator",
    forward_height=4,
    reverse_height=4,
    prop_layers=[3],
    forward_prop="residual",
    forward_kwargs={"conv_name":'sepconv', 'se_block':True},
    mid_prop="residual",
    mid_kwargs={"conv_name":'sepconv', 'se_block':True},
    reverse_prop="residual",
    reverse_kwargs={"conv_name":'sepconv', 'se_block':True},
    forward_conv_name="sepconv",
    reverse_conv_name="sepconv",
    pooler='depthwise_conv',
    unet_attention=True,
    channels_per_step=2,
  )

CONV_FNS = ["conv", "depthconv", "sepconv", "conv2dn"]
PROPAGATORS = ["vanilla", "residual", "attention", "dense"]
POOLERS = ['maxpool', 'depthwise_conv', 'conv', "residual", "xception"]

class Conv2DN(Layer):

  def __init__(self, filters, kernel_size=[1, 1], padding="same",
    kernel_initializer='he_normal', activation=None):
    super(Conv2DN, self).__init__()
    self.conv = Conv2D(filters, kernel_size, padding=padding,
      kernel_initializer = kernel_initializer)
    self.norm = BatchNormalization()
    if activation:
      self.activation = Activation(activation)
    else:
      self.activation = None

  def call(self, inputs):
    res = self.conv(inputs)
    res = self.norm(res)
    if self.activation:
      res = self.activation(res)

    return res


def get_conv_fn(name):
  assert name in CONV_FNS, "Choices must be in {}".format(CONV_FNS)

  if name == "conv":
    return Conv2D
  elif name == "depthconv":
    return DepthwiseConv2D
  elif name == "sepconv":
    return SeparableConv2D
  else:
    return Conv2DN


def additive_attention(query, key, intermediate_filters=64,
  name='add_attention'):
  '''
  Computes additive attention coefficients given query and key.

  Arguments:
    query: `tf.Tensor` with shape [B, H, W, C].
    key: `tf.Tensor` with shape [B, H, W, C'].
    intermediate_filters: int denoting channel dimension to project
      query and key.

  Returns:
    `tf.Tensor` with shape [B, H, W] representing attention coefficients.
  '''

  with tf.name_scope(name):
    proj_q = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(query)
    proj_k = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(key)
    added = Add().apply([proj_q, proj_k])
    relued = ReLU().apply(added)
    res = Conv2D(1, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(relued)
    res = Activation('sigmoid').apply(res)
    return res


def squeeze_excitation(input, ratio=4, name="squeeze_excitation"):
  '''
  Modifies input along channels based on the squeeze-excitation procedure.

  Arguments:
    input: `tf.Tensor` with shape [B, H, W, C].
    ratio: int denoting intermediate channels to use for computation. ratio must
      divide C above.

  Returns:
    input: `tf.Tensor` with shape [B, H, W, C].
  '''
  with tf.name_scope(name):
    filters = int(input.shape[-1])
    squeeze = GlobalAveragePooling2D().apply(input)
    excitation = Dense(filters // ratio, activation='relu',
      kernel_initializer='he_normal').apply(squeeze)
    excitation = Dense(filters, activation='sigmoid').apply(excitation)
    excitation = Reshape((1, 1, filters)).apply(excitation)
    scaled = Lambda(lambda x: x * excitation).apply(input)
    return scaled


class ObsAttention(Layer):

  def __init__(self, intermediate_filters, kernel_size=[5, 5]):
    super(ObsAttention, self).__init__()
    self.conv_1 = Conv2D(intermediate_filters, kernel_size, padding="same", activation="relu",
      kernel_initializer = 'he_normal')
    self.conv_2 = Conv2D(1, [1, 1], padding="same", activation='sigmoid')

  def call(self, inputs):
    [estimate, truth] = inputs
    proj = self.conv_1(estimate-truth)
    res = self.conv_2(proj)
    return res

class AddAttention(Layer):

  def __init__(self, intermediate_filters):
    super(AddAttention, self).__init__()
    self.proj_q = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal')
    self.proj_k = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal')
    self.relu = ReLU()
    self.post_conv = Conv2D(1, [1, 1], padding="same", activation='sigmoid')

  def call(self, inputs):
    [query, key] = inputs
    proj_q = self.proj_q(query)
    proj_k = self.proj_k(key)
    relued = self.relu(proj_q + proj_k)
    res = self.post_conv(relued)
    return res


class ResizeImage(Layer):
  """
  Layer to resize images via tf.image.ResizeMethod
  """

  def __init__(self, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    align_corners=True):
    super(ResizeImage, self).__init__()
    self.new_shape = new_shape
    self.method = method
    self.align_corners = align_corners

  def call(self, input):
    return tf.image.resize_images(input, self.new_shape, method=self.method,
      align_corners=self.align_corners)


def vanilla_propagator(input, conv_name, se_block=False, kernel_size=[3, 3],
  name="vanilla_prop"):
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    filters = int(input.shape[-1])
    res = conv_fn(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(input)
    res = conv_fn(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = squeeze_excitation(res)
    return res


def res_propagator(input, conv_name, se_block=False, kernel_size=[3, 3],
  name="res_prop"):
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    filters = int(input.shape[-1])
    res = BatchNormalization().apply(input)
    res = ReLU().apply(res)
    res = conv_fn(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = BatchNormalization().apply(res)
    res = ReLU().apply(res)
    res = conv_fn(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = squeeze_excitation(res)
    res = Add().apply([res, input])
    return res


def attention_propagator(input, layers=2, se_block=False,
  name="attention_prop"):
  with tf.variable_scope(name):
    filters = int(input.shape[-1])
    hparams = tf.contrib.training.HParams(
      hidden_size=filters,
      layer_prepostprocess_dropout=0,
      num_heads=4,
      query_shape=(8,8),
      memory_flange=(8,8),
      pos="timing",
      norm_type = "layer",
      norm_epsilon=1e-6,
      layer_preprocess_sequence = "n",
      layer_postprocess_sequence="da",
      attention_key_channels = 0,
      attention_value_channels = 0,
      ffn_layer="conv_hidden_relu",
      filter_size = 128,
      relu_dropout=0.0,
    )
    def attention_(input):
      res = cia.prepare_encoder(input, hparams, "local_2d")
      res = cia.transformer_encoder_layers(
          res,
          layers,
          hparams,
          attention_type=cia.AttentionType.LOCAL_2D,
          name="encoder")
      return res

    res = Lambda(attention_).apply(input)
    return res


def dense_propagator(input, conv_name, layers=3, kernel_size=[3, 3],
  name="dense_prop"):
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    filters = int(input.shape[-1])
    concated = input
    for _ in range(layers):
      conved = conv_fn(filters, kernel_size, padding="same",
        kernel_initializer='he_normal', activation='relu').apply(concated)
      concated = Concatenate().apply([concated, conved])

    return conved


def get_propagator(name):
  assert name in PROPAGATORS, "Propagator must be in {}".format(PROPAGATORS)

  if name == "vanilla":
    return vanilla_propagator
  elif name == "residual":
    return res_propagator
  elif name == "attention":
    return attention_propagator
  else:
    return dense_propagator


def get_embedding(input, embedding_string, filters, name="embedding"):
  with tf.name_scope(name):
    res = input
    for block in embedding_string.split('/'):
      if '_' in block:
        fn_name = block.split('_')[0]
        number = int(block.split('_')[1])
      else:
        fn_name = block
        number = 1

      if 'conv' in fn_name:
        conv_fn = get_conv_fn(fn_name)
        for _ in range(number):
          res = conv_fn(filters, [1, 1], padding="same",
            kernel_initializer = 'he_normal').apply(res)
      elif fn_name == 'attention':
        for i in range(1, number+1):
          res = attention_propagator(res, name=name+'_attention_{}'.format(i))
      elif fn_name == 'se':
        for _ in range(number):
          res = squeeze_excitation(res)
      elif fn_name == 'xception' or fn_name == 'residual':
        if fn_name == 'xception':
          conv_name = 'sepconv'
        else:
          conv_name = 'conv'
        for _ in range(number):
          pooled_filters = int(res.shape[-1]) * 2
          res = perform_pooling(res, pooled_filters, fn_name,
            conv_name=conv_name)
    return res

def build_model_from_propagator(input, hparams, num_classes,
  name='propagator_model'):
  with tf.name_scope(name):
    prop_fn = get_propagator(hparams.forward_prop)

    res = input
    for i, elem in enumerate(hparams.prop_layers):
      for j in range(1, elem+1):
        res = prop_fn(res, name=hparams.forward_prop + '_{}_{}'.format(i+1, j),
          **hparams.forward_kwargs)
      if i < len(hparams.prop_layers) - 1:
        pooled_filters = int(res.shape[-1]) * 2
        res = perform_pooling(res, pooled_filters, hparams.pooler,
          conv_name=hparams.forward_conv_name, name='pooling_{}'.format(i+1))

    res = Dropout(rate=hparams.dropout).apply(res)
    res = Conv2D(num_classes, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(res)

    return Model(inputs=input, outputs=res)


def xception_pooler(input, pooled_filters, conv_name='sepconv',
  kernel_size=[3, 3], name="xception_pooler"):
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    res = conv_fn(pooled_filters, kernel_size, activation='relu',
      padding="same", kernel_initializer = 'he_normal').apply(input)
    res = conv_fn(pooled_filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = MaxPool2D(pool_size=(2, 2)).apply(res)
    projected = Conv2D(pooled_filters, kernel_size=[1, 1], strides=[2, 2],
      padding="same").apply(input)
    return Add().apply([res, projected])


def resnet_pooler(input, pooled_filters, conv_name='conv', kernel_size=[3, 3]
  , name="resnet_pooler"):
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    res = conv_fn(pooled_filters, kernel_size, activation='relu',
      padding="same", strides=[2, 2],
      kernel_initializer = 'he_normal').apply(input)
    res = conv_fn(pooled_filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    projected = Conv2D(pooled_filters, kernel_size=[1, 1], strides=[2, 2],
      padding="same").apply(input)
    return Add().apply([res, projected])

def perform_pooling(input, pooled_filters, pooler, conv_name='conv',
  kernel_size=[3, 3], name='pooling'):

  assert pooler in POOLERS, 'pooler must be in {}'.format(POOLERS)

  with tf.name_scope(name):
    if pooler == 'conv':
      conv_fn = get_conv_fn(conv_name)
      pooled = conv_fn(pooled_filters, kernel_size, activation=None,
        strides=[2,2], padding="same").apply(input)
    elif pooler == 'depthwise_conv':
      pooled = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=2,
        kernel_size=kernel_size,
        padding="same",
        strides=2,
      ).apply(input)
      pooled = BatchNormalization().apply(pooled)
    elif pooler == 'maxpool':
      conv_fn = get_conv_fn(conv_name)
      pooled = conv_fn(pooled_filters, kernel_size, activation='relu',
        padding="same").apply(input)
      pooled = MaxPool2D([2, 2]).apply(pooled)
    elif pooler == 'xception':
      return xception_pooler(input, pooled_filters, conv_name=conv_name,
        kernel_size=kernel_size)
    else:
      return resnet_pooler(input, pooled_filters, conv_name=conv_name,
        kernel_size=kernel_size)

def down_sampling_block(input, pooled_filters, propagator, prop_kwargs,
  conv_name='conv', kernel_size=[3, 3], activation='relu', pooler='maxpool',
  name="down_sample"):

  with tf.name_scope(name):
    res = propagator(input, **prop_kwargs)
    pooled = perform_pooling(res, pooled_filters, pooler, conv_name=conv_name,
      kernel_size=kernel_size)

    return res, pooled


def conv_concat(input, skip_input, filters, conv_fn=Conv2D,
  up_kernel_size=[2, 2], kernel_size=[3, 3], attention=True, name='conv_concat'):
  with tf.name_scope(name):
    res = ResizeImage(skip_input.shape[1:3]).apply(input)
    res = conv_fn(int(skip_input.shape[-1]), up_kernel_size,
      padding="same", activation='relu').apply(res)
    if attention:
      coefficients = additive_attention(skip_input, res)
      skip_input = Lambda(lambda x: x * coefficients).apply(skip_input)
    res = Concatenate().apply([res, skip_input])
    res = conv_fn(filters, kernel_size, padding="same",
      activation='relu').apply(res)
    return res


def up_sampling_block(input, skip_input, filters, propagator, prop_kwargs,
  conv_name='conv', attention=True, up_kernel_size=[2, 2], kernel_size=[3, 3],
  name="up_sample"):

  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    res = conv_concat(input, skip_input, filters, conv_fn, up_kernel_size,
      kernel_size, attention)
    res = propagator(res, **prop_kwargs)
    return res


def build_unet_from_propagator(input, hparams, num_classes, name='unet'):
  with tf.name_scope(name):
    pooled = input
    skip_inputs = []
    forward_prop_fn = get_propagator(hparams.forward_prop)
    initial_filters = int(input.shape[-1])
    for i in range(1, hparams.forward_height+1):
      conved, pooled = down_sampling_block(pooled, initial_filters * (2 ** i),
        forward_prop_fn, hparams.forward_kwargs,
        conv_name=hparams.forward_conv_name, pooler=hparams.pooler,
        name="down_sample_{}".format(i))
      if i > hparams.forward_height - hparams.reverse_height:
        skip_inputs.append(conved)

    mid_prop_fn = get_propagator(hparams.mid_prop)
    res = mid_prop_fn(pooled, name='mid_prop', **hparams.mid_kwargs)

    reverse_prop_fn = get_propagator(hparams.reverse_prop)
    for i in range(1, hparams.reverse_height+1):
      res = up_sampling_block(res, skip_inputs[hparams.reverse_height - i],
        initial_filters * (2 ** (hparams.forward_height-i)),
        reverse_prop_fn, hparams.reverse_kwargs,
        conv_name=hparams.reverse_conv_name,
        attention=hparams.unet_attention, name="up_sample_{}".format(i))

    res = Dropout(rate=hparams.dropout).apply(res)
    res = Conv2D(num_classes, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(res)

    return Model(inputs=input, outputs=res)


def recurrent_net_v1(input, num_classes, hparams, initial_inv_hidden=None,
  initial_inv_cell=None, name="recurrent_unet_v1"):
  '''
  Recurrent net that learns both observations and distributions
  '''
  assert(input.shape[-1] % hparams.channels_per_step == 0), \
    "Channels per timestep must divide number of observations."
  assert('xception' not in hparams.embedding and 'residual' not in
    hparams.embedding), "Downsampling embeddings not allowed when recurring."
  if hparams.model_type == 'unet':
    assert(hparams.forward_height == hparams.reverse_height), "Forward and \
      reverse heights must be identical in recurrent architecture."

  with tf.name_scope(name):
    one_hot_zeros = tf.zeros([tf.shape(input)[0], input.shape[1],
      input.shape[2]], dtype=tf.int32)
    one_hot_zeros = tf.one_hot(one_hot_zeros, num_classes)
    if initial_inv_hidden is None:
      initial_inv_estimate = one_hot_zeros
    if initial_inv_cell is None:
      initial_inv_cell = one_hot_zeros

    split_size = hparams.channels_per_step
    num_split = int(input.shape[-1]) // split_size

    inv_estimates = []
    obs_estimates = []
    prev_inv_estimate = initial_inv_estimate
    prev_inv_cell = initial_inv_cell

    shape = input.shape
    example = Input(tensor=tf.zeros([tf.shape(input)[0], int(shape[1]),
      int(shape[2]), hparams.initial_filters]))

    if hparams.model_type == 'unet':
      forward_net = build_unet_from_propagator(example, hparams,
        split_size, name="forward_unet")
      inverse_net = build_unet_from_propagator(example, hparams,
        num_classes, name="inverse_unet")
    else:
      forward_net = build_model_from_propagator(example, hparams,
        split_size, name="forward_net")
      inverse_net = build_model_from_propagator(example, hparams,
        split_size, name="inverse_net")

    if hparams.gating == 'LSTM':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)
      output_gate = ObsAttention(32)
    elif hparams.gating == 'GRU':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)

    for i, obs in enumerate(tf.split(input, num_split, -1)):
      embedded_inv = get_embedding(prev_inv_estimate, hparams.embedding,
        hparams.initial_filters, name="inv_embedding_{}".format(i+1))
      obs_estimate = forward_net(embedded_inv)
      obs_estimates.append(obs_estimate)

      if hparams.gating == 'LSTM':
        forget_coefficients = forget_gate([obs_estimate, obs])
        input_coefficients = input_gate([obs_estimate, obs])
        output_coefficients = output_gate([obs_estimate, obs])
      elif hparams.gating == 'GRU':
        forget_coefficients = forget_gate([obs_estimate, obs])
        input_coefficients = input_gate([obs_estimate, obs])
        output_coefficients =  1
      else:
        forget_coefficients = input_coefficients = output_coefficients =  1

      concated = tf.concat([obs_estimate, obs], -1)
      embedded_concated = get_embedding(concated, hparams.embedding,
        hparams.initial_filters, name="obs_embedding_{}".format(i+1))
      inverse_shift = inverse_net(embedded_concated)

      prev_inv_cell = input_coefficients * inverse_shift + forget_coefficients * prev_inv_cell
      prev_inv_estimate = output_coefficients * prev_inv_cell

      inv_estimates.append(prev_inv_estimate)

    return obs_estimates, inv_estimates


def recurrent_net_v2(input, num_classes, hparams, initial_inv_hidden=None,
  initial_inv_cell=None, name="recurrent_unet_v1"):
  '''
  Recurrent net that learns distributions
  '''
  assert(input.shape[-1] % hparams.channels_per_step == 0), \
    "Channels per timestep must divide number of observations"
  assert('xception' not in hparams.embedding and 'residual' not in
    hparams.embedding), "Downsampling embeddings not allowed when recurring."

  with tf.name_scope(name):
    one_hot_zeros = tf.zeros([tf.shape(input)[0], input.shape[1],
      input.shape[2]], dtype=tf.int32)
    one_hot_zeros = tf.one_hot(one_hot_zeros, num_classes)
    if initial_inv_hidden is None:
      initial_inv_estimate = one_hot_zeros
    if initial_inv_cell is None:
      initial_inv_cell = one_hot_zeros

    split_size = hparams.channels_per_step
    num_split = int(input.shape[-1]) // split_size

    inv_estimates = []
    prev_inv_estimate = initial_inv_estimate
    prev_inv_cell = initial_inv_cell

    shape = input.shape
    example = Input(tensor=tf.zeros([tf.shape(input)[0], int(shape[1]),
      int(shape[2]), hparams.initial_filters]))

    if hparams.model_type == 'unet':
      net = build_unet_from_propagator(example, hparams, num_classes,
        name="inverse_unet")
    else:
      net = build_model_from_propagator(example, hparams, split_size,
        name="inverse_net")

    if hparams.gating == 'LSTM':
      forget_gate = AddAttention(32)
      input_gate = AddAttention(32)
      output_gate = AddAttention(32)
    elif hparams.gating == 'GRU':
      forget_gate = AddAttention(32)
      input_gate = AddAttention(32)

    for i, obs in enumerate(tf.split(input, num_split, -1)):
      if hparams.gating == 'LSTM':
        forget_coefficients = forget_gate([prev_inv_estimate, obs])
        input_coefficients = input_gate([prev_inv_estimate, obs])
        output_coefficients = output_gate([prev_inv_estimate, obs])
      elif hparams.gating == 'GRU':
        forget_coefficients = forget_gate([prev_inv_estimate, obs])
        input_coefficients = input_gate([prev_inv_estimate, obs])
        output_coefficients =  1
      else:
        forget_coefficients = input_coefficients = output_coefficients =  1

      concated = tf.concat([prev_inv_estimate, obs], -1)
      embedded_concated = get_embedding(concated, hparams.embedding,
        hparams.initial_filters, name="embedding_{}".format(i+1))
      shift = net(embedded_concated)
      prev_inv_cell = input_coefficients * shift + forget_coefficients * prev_inv_cell
      prev_inv_estimate = output_coefficients * prev_inv_cell

      inv_estimates.append(prev_inv_estimate)

    return None, inv_estimates


def gpu_preprocess(observations, distributions, params):

  distributions, observations = preprocess.hilbert(hilbert_axis=3)(distributions, observations)

  num_angles = len(params.angle_indices)
  num_freqs = len(params.frequency_indices)

  distributions = distributions[ ..., tf.newaxis]
  distributions = tf.keras.layers.AveragePooling2D(
    params.distribution_pool_downsample).apply(distributions) * (
      params.distribution_pool_downsample ** 2)
  distributions = distributions[..., 0]

  angles = params.observation_spec.angles

  observation_pooling_layer = tf.keras.layers.AveragePooling2D(
    params.observation_pool_downsample)

  storage = []
  for freqs, ang in zip(tf.split(observations, observations.shape[1], 1), angles):
    pooled = observation_pooling_layer.apply(tf.squeeze(freqs, 1))
    height = int(pooled.shape[1])
    width = int(pooled.shape[2])
    rotated = tf.contrib.image.rotate(pooled, -1 * ang, interpolation='BILINEAR')
    storage.append(rotated)

  if len(storage) > 1:
    observations = tf.keras.layers.Concatenate(axis=-1).apply(storage)
  else:
    observations = storage[0]

  observations.set_shape([None, height, width, num_angles * num_freqs])


  return observations, distributions


def model_fn(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: dict containing:
      `images`: a `tf.Tensor` with shape `[batch_size, height, width, channels]`
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

  tf.summary.image("original_distribution", distributions[..., tf.newaxis], 1)

  observations, distributions = gpu_preprocess(observations, distributions, params)

  logging.info("`observations` tensor after gpu preprocess in model is "
                "{}".format(observations))
  logging.info("`distributions` tensor  after gpu preprocess in model is "
                "{}".format(distributions))

  distributions_values = loss_utils.quantize_tensor(
    distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth, False)
  distributions_quantized = tf.one_hot(distributions_values, 2 ** params.bit_depth)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, ang_index in enumerate(params.angle_indices):
      angle = params.observation_spec.angles[ang_index]
      tf.summary.image("obs_angle_{}".format(angle), observations[..., i, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)

  # embedded = Conv2D(params.initial_filters, [1, 1], activation=None,
  #   padding="same", kernel_initializer = 'he_normal').apply(observations)
  # if mode ==  tf.estimator.ModeKeys.TRAIN:
  #   training = True
  # else:
  #   training = False
  # training = tf.constant(training, dtype=tf.bool)
  embedded = get_embedding(observations, params.embedding,
    params.initial_filters)
  input = Input(tensor=embedded)

  if params.model_type == 'unet':
    model = build_unet_from_propagator(input, params, 2 ** params.bit_depth)
    predictions_quantized = model(embedded)
  else:
    model = build_model_from_propagator(input, params, 2 ** params.bit_depth)
    predictions_quantized = model(embedded)


  logging.info("predictions_quantized {}".format(predictions_quantized))
  logging.info("distributions_quantized {}".format(distributions_quantized))


  with tf.variable_scope("predictions"):
    def _logit_to_class(logit):
      return tf.argmax(logit, -1)
    distribution_class = _logit_to_class(distributions_quantized)
    prediction_class = _logit_to_class(predictions_quantized)

    # Visualize output of predictions as categories.
    tf.summary.tensor_summary("prediction_class", prediction_class)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class), tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class), tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    tf.summary.scalar("nonzero_fraction", nonzero_fraction)
    nonzero_hook = tf.train.LoggingTensorHook(
      tensors={
        "predicted_nonzero_fraction": nonzero_fraction,
        "true_nonzero_fraction": true_nonzero_fraction,
      },
      every_n_iter=50,
    )
    hooks.append(nonzero_hook)


    def _class_to_image(category):
      return tf.cast(category, tf.float32)[..., tf.newaxis]
    dist_image = _class_to_image(distribution_class)
    pred_image = _class_to_image(prediction_class)

    image_hook = tf.train.LoggingTensorHook(
      tensors={"distribution": dist_image[0, ..., 0],
               "prediction": pred_image[0, ..., 0],},
      every_n_iter=50,
    )
    hooks.append(image_hook)

    tf.summary.image("distributions", dist_image, 1)
    tf.summary.image("predictions", pred_image, 1)
    tf.summary.image("difference", (dist_image - pred_image) ** 2, 1)

    # Visualize output of predictions as categories.
    dist_summary = tf.summary.tensor_summary("distribution_tensor", dist_image)
    pred_summary = tf.summary.tensor_summary("predictions_tensor", pred_image)
    diff_summary = tf.summary.tensor_summary("difference_tensor", dist_image-pred_image)

    predict_output = {
        "predictions": pred_image
    }

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    proportion = (tf.reduce_sum(
        distributions_quantized,
        axis=[0, 1, 2],
        keepdims=True,
        ) + 2 ** params.bit_depth) / (tf.reduce_sum(distributions_quantized) + 2 ** params.bit_depth)

    # proportion = (tf.reduce_sum(
    #     distributions_quantized,
    #     axis=[0, 1, 2],
    #     keepdims=True,
    #     ) + 2 ** params.bit_depth)
    inv_proportion = 1 / proportion

    ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)

    def bets_and_rewards_fn(params):
      if params.rewards == "1/n":
        rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
      elif params.rewards == "-logn":
        rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) * distributions_quantized,
          axis=-1)
      elif params.rewards == "log1/n":
        rewards = tf.reduce_sum(tf.math.log(inv_proportion) * distributions_quantized,
          axis=-1)
      elif params.rewards == "1/sqrtn":
        rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1))
      else:
        rewards = ones_like

      one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)

      if params.bets == "1/n":
        bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
      elif params.bets== "-logn":
        bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
          axis=-1)
      elif params.bets == "log1/n":
        bets = tf.reduce_sum(tf.math.log(inv_proportion) * one_hot_predictions,
          axis=-1)
      elif params.bets == "1/sqrtn":
        bets = tf.math.sqrt(tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1))
      else:
        bets = ones_like

      return bets, rewards


    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
      lambda: (ones_like, ones_like))
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportional_weights = bets * rewards
    if params.diff_scale == "abs":
      proportional_weights *= tf.cast(tf.math.abs(distribution_class - prediction_class), tf.float32)
    elif params.diff_scale == "square":
      proportional_weights *= tf.cast(tf.math.square(distribution_class - prediction_class), tf.float32)

    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0], "log_inv": tf.math.log(inv_proportion)},
      every_n_iter=50,
    )
    hooks.append(proportion_hook)

    softmax_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=distributions_quantized,
      logits=predictions_quantized,
      weights=proportional_weights
    )

    tf.summary.scalar("softmax_loss", softmax_loss)

    loss = softmax_loss

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=lr,
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

    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)

    proportional_weights = loss_utils.inverse_class_weight(distributions_quantized)

    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy,},
      every_n_iter=50
    )
    hooks.append(accuracy_hook)

    accuracy = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    accuracy_weighted = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1),
      weights=proportional_weights,
    )

    precision = tf.metrics.precision(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    recall = tf.metrics.recall(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    f1 = tf.where(tf.equal(precision[0] + recall[0], 0.),
      tf.constant(0, dtype=tf.float32), 2 * precision[0] * recall[0] / (precision[0] + recall[0]))

    labels = tf.argmax(distributions_quantized, -1)
    predictions= tf.argmax(predictions_quantized, -1)
    non_zero = tf.where(tf.equal(0, tf.cast(labels, dtype=tf.int32)), -1 * tf.ones_like(labels), labels)
    non_zero_correct = tf.math.reduce_sum(tf.cast(tf.equal(non_zero, predictions), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0, tf.cast(labels, dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0), tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

    tf.summary.scalar("non_zero_acc", non_zero_acc)
    tf.summary.scalar("precision", precision[0])
    tf.summary.scalar("recall", recall[0])
    tf.summary.scalar("f1", f1)


    eval_metric_ops = {
      "accuracy": accuracy,
      "accuracy_weighted": accuracy_weighted,
      "precision": precision,
      "recall": recall,
      "f1": tf.metrics.mean(f1),
      "non_zero_acc": tf.metrics.mean(non_zero_acc),
    }

    merged = tf.summary.merge([dist_summary, pred_summary, diff_summary])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=hooks,
    evaluation_hooks=[tf.train.SummarySaverHook(save_steps=1, output_dir= params.job_dir + "/eval", summary_op = merged)]
  )


def model_fn_original(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: dict containing:
      `images`: a `tf.Tensor` with shape `[batch_size, height, width, channels]`
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

  tf.summary.image("original_distribution", distributions[..., tf.newaxis], 1)

  observations, distributions = gpu_preprocess(observations, distributions, params)

  logging.info("`observations` tensor after gpu preprocess in model is "
                "{}".format(observations))
  logging.info("`distributions` tensor  after gpu preprocess in model is "
                "{}".format(distributions))

  distributions_values = loss_utils.quantize_tensor(
    distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth, False)
  distributions_quantized = tf.one_hot(distributions_values, 2 ** params.bit_depth)

  observations_hook = tf.train.LoggingTensorHook(
      tensors={
        "obs_max": tf.math.reduce_max(observations),
        "obs_min": tf.math.reduce_min(observations),
        "dis_max": tf.math.reduce_max(distributions),
        "dis_min": tf.math.reduce_min(distributions)
      },
      every_n_iter=50,
  )
  hooks.append(observations_hook)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, ang_index in enumerate(params.angle_indices):
      angle = params.observation_spec.angles[ang_index]
      tf.summary.image("obs_angle_{}".format(angle), observations[..., i, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)

  # embedded = Conv2D(params.initial_filters, [1, 1], activation=None,
  #   padding="same", kernel_initializer = 'he_normal').apply(observations)
  if mode ==  tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False
  training = tf.constant(training, dtype=tf.bool)

  embedded = get_embedding(observations, params.embedding, params.initial_filters)

  if params.unet_type == 'residual':
    unet_output = unet_res(embedded, params.unet_height, params.initial_filters,
      training, dropout_rate=params.dropout)
    predictions_quantized = tf.keras.layers.Conv2D(
      filters=2 ** params.bit_depth,
      kernel_size=[1, 1],
      padding="same",
    ).apply(unet_output)
  elif params.unet_type == 'attention_net':
    predictions_quantized = self_attention_net(embedded, params.initial_filters, 8, 2**params.bit_depth)
  elif params.unet_type == 'se_res_net':
    predictions_quantized = se_res_net(embedded, 2**params.bit_depth, params.se_blocks)
  elif params.unet_type == 'dense_net':
    model = DenseNetBlock(params.initial_filters)
    output = model(embedded)
    predictions_quantized = tf.keras.layers.Conv2D(
      filters=2 ** params.bit_depth,
      kernel_size=[1, 1],
      padding="same",
    ).apply(output)
  else:
    input = Input(embedded.shape[1:])
    model = unet(input, params.unet_height, params.initial_filters, 2**params.bit_depth,
      dropout_rate=params.dropout, type=params.unet_type)
    predictions_quantized = model(embedded)

  logging.info("predictions_quantized {}".format(predictions_quantized))
  logging.info("distributions_quantized {}".format(distributions_quantized))


  with tf.variable_scope("predictions"):
    def _logit_to_class(logit):
      return tf.argmax(logit, -1)
    distribution_class = _logit_to_class(distributions_quantized)
    prediction_class = _logit_to_class(predictions_quantized)

    # Visualize output of predictions as categories.
    tf.summary.tensor_summary("prediction_class", prediction_class)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class), tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class), tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    tf.summary.scalar("nonzero_fraction", nonzero_fraction)
    nonzero_hook = tf.train.LoggingTensorHook(
      tensors={
        "predicted_nonzero_fraction": nonzero_fraction,
        "true_nonzero_fraction": true_nonzero_fraction,
      },
      every_n_iter=50,
    )
    hooks.append(nonzero_hook)


    def _class_to_image(category):
      return tf.cast(category, tf.float32)[..., tf.newaxis]
    dist_image = _class_to_image(distribution_class)
    pred_image = _class_to_image(prediction_class)

    image_hook = tf.train.LoggingTensorHook(
      tensors={"distribution": dist_image[0, ..., 0],
               "prediction": pred_image[0, ..., 0],},
      every_n_iter=50,
    )
    hooks.append(image_hook)

    tf.summary.image("distributions", dist_image, 1)
    tf.summary.image("predictions", pred_image, 1)
    tf.summary.image("difference", (dist_image - pred_image) ** 2, 1)

    # Visualize output of predictions as categories.
    dist_summary = tf.summary.tensor_summary("distribution_tensor", dist_image)
    pred_summary = tf.summary.tensor_summary("predictions_tensor", pred_image)
    diff_summary = tf.summary.tensor_summary("difference_tensor", dist_image-pred_image)

    predict_output = {
        "predictions": pred_image
    }

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    proportion = (tf.reduce_sum(
        distributions_quantized,
        axis=[0, 1, 2],
        keepdims=True,
        ) + 2 ** params.bit_depth) / (tf.reduce_sum(distributions_quantized) + 2 ** params.bit_depth)

    # proportion = (tf.reduce_sum(
    #     distributions_quantized,
    #     axis=[0, 1, 2],
    #     keepdims=True,
    #     ) + 2 ** params.bit_depth)
    inv_proportion = 1 / proportion

    ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)

    def bets_and_rewards_fn(params):
      if params.rewards == "1/n":
        rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
      elif params.rewards == "-logn":
        rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) * distributions_quantized,
          axis=-1)
      elif params.rewards == "log1/n":
        rewards = tf.reduce_sum(tf.math.log(inv_proportion) * distributions_quantized,
          axis=-1)
      elif params.rewards == "1/sqrtn":
        rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1))
      else:
        rewards = ones_like

      one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)

      if params.bets == "1/n":
        bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
      elif params.bets== "-logn":
        bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
          axis=-1)
      elif params.bets == "log1/n":
        bets = tf.reduce_sum(tf.math.log(inv_proportion) * one_hot_predictions,
          axis=-1)
      elif params.bets == "1/sqrtn":
        bets = tf.math.sqrt(tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1))
      else:
        bets = ones_like

      return bets, rewards


    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
      lambda: (ones_like, ones_like))
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportional_weights = bets * rewards
    if params.diff_scale == "abs":
      proportional_weights *= tf.cast(tf.math.abs(distribution_class - prediction_class), tf.float32)
    elif params.diff_scale == "square":
      proportional_weights *= tf.cast(tf.math.square(distribution_class - prediction_class), tf.float32)

    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0], "log_inv": tf.math.log(inv_proportion)},
      every_n_iter=50,
    )
    hooks.append(proportion_hook)

    softmax_loss = tf.losses.softmax_cross_entropy(
      onehot_labels=distributions_quantized,
      logits=predictions_quantized,
      weights=proportional_weights
    )

    tf.summary.scalar("softmax_loss", softmax_loss)

    loss = softmax_loss

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=lr,
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

    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)

    proportional_weights = loss_utils.inverse_class_weight(distributions_quantized)

    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy,},
      every_n_iter=50
    )
    hooks.append(accuracy_hook)

    accuracy = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    accuracy_weighted = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1),
      weights=proportional_weights,
    )

    precision = tf.metrics.precision(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    recall = tf.metrics.recall(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    f1 = tf.where(tf.equal(precision[0] + recall[0], 0.),
      tf.constant(0, dtype=tf.float32), 2 * precision[0] * recall[0] / (precision[0] + recall[0]))

    labels = tf.argmax(distributions_quantized, -1)
    predictions= tf.argmax(predictions_quantized, -1)
    non_zero = tf.where(tf.equal(0, tf.cast(labels, dtype=tf.int32)), -1 * tf.ones_like(labels), labels)
    non_zero_correct = tf.math.reduce_sum(tf.cast(tf.equal(non_zero, predictions), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0, tf.cast(labels, dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0), tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

    tf.summary.scalar("non_zero_acc", non_zero_acc)
    tf.summary.scalar("precision", precision[0])
    tf.summary.scalar("recall", recall[0])
    tf.summary.scalar("f1", f1)


    eval_metric_ops = {
      "accuracy": accuracy,
      "accuracy_weighted": accuracy_weighted,
      "precision": precision,
      "recall": recall,
      "f1": tf.metrics.mean(f1),
      "non_zero_acc": tf.metrics.mean(non_zero_acc),
    }

    merged = tf.summary.merge([dist_summary, pred_summary, diff_summary])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=hooks,
    evaluation_hooks=[tf.train.SummarySaverHook(save_steps=1, output_dir= params.job_dir + "/eval", summary_op = merged)]
  )

def model_fn_recurrent(features, labels, mode, params):
  """Defines model graph for super resolution.

  Args:
    features: dict containing:
      `images`: a `tf.Tensor` with shape `[batch_size, height, width, channels]`
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

  tf.summary.image("original_distribution", distributions[..., tf.newaxis], 1)

  observations, distributions = gpu_preprocess(observations, distributions, params)

  logging.info("`observations` tensor after gpu preprocess in model is "
                "{}".format(observations))
  logging.info("`distributions` tensor  after gpu preprocess in model is "
                "{}".format(distributions))

  distributions_values = loss_utils.quantize_tensor(
    distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth, False)
  distributions_quantized = tf.one_hot(distributions_values, 2 ** params.bit_depth)

  observations_hook = tf.train.LoggingTensorHook(
      tensors={
        "obs_max": tf.math.reduce_max(observations),
        "obs_min": tf.math.reduce_min(observations),
        "dis_max": tf.math.reduce_max(distributions),
        "dis_min": tf.math.reduce_min(distributions)
      },
      every_n_iter=50,
  )
  hooks.append(observations_hook)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, ang_index in enumerate(params.angle_indices):
      angle = params.observation_spec.angles[ang_index]
      tf.summary.image("obs_angle_{}".format(angle), observations[..., i, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)

  if mode ==  tf.estimator.ModeKeys.TRAIN:
    training = True
  else:
    training = False

  if params.recurrent == "recurrent":
    obs_estimates, inv_estimates = recurrent_unet(observations, 2 ** params.bit_depth,
      params.unet_height, params.unet_height, params.initial_filters, params.initial_filters,
      params.dropout, params.dropout, training, unet_type=params.unet_type)
  elif params.recurrent == "recurrent_v1":
    obs_estimates, inv_estimates = recurrent_net_v1(observations,
      2 ** params.bit_depth, params)
  elif params.recurrent == "recurrent_v2":
    obs_estimates, inv_estimates = recurrent_net_v2(observations, 2 ** params.bit_depth,
      params)
  elif params.recurrent == "recurrent_v3":
    obs_estimates, inv_estimates = recurrent_unet_v3(observations, 2 ** params.bit_depth,
      params.unet_height, params.unet_height, params.initial_filters, params.initial_filters,
      params.dropout, params.dropout, training, params.num_split, unet_type=params.unet_type)
  elif params.recurrent == "recurrent_v4":
    obs_estimates, inv_estimates = recurrent_unet_v4(observations, 2 ** params.bit_depth,
      params.unet_height, params.unet_height, params.initial_filters, params.initial_filters,
      params.dropout, params.dropout, training, params.num_split, gating=params.gating,
      unet_type=params.unet_type, embedding=params.embedding)
  elif params.recurrent == "bidirectional_recurrent_v4":
    obs_estimates, inv_estimates = bidirectional_recurrent_unet_v4(observations, 2 ** params.bit_depth,
      params.unet_height, params.unet_height, params.initial_filters, params.initial_filters,
      params.dropout, params.dropout, training, params.num_split, unet_type=params.unet_type)


  logging.info("predictions_quantized {}".format(inv_estimates[-1]))
  logging.info("distributions_quantized {}".format(distributions_quantized))


  with tf.variable_scope("predictions"):
    def _logit_to_class(logit):
      return tf.argmax(logit, -1)
    distribution_class = _logit_to_class(distributions_quantized)
    prediction_classes = [_logit_to_class(pred) for pred in inv_estimates]
    prediction_class = prediction_classes[-1]

    # Visualize output of predictions as categories.
    tf.summary.tensor_summary("prediction_class", prediction_class)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class), tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class), tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(prediction_class), tf.float32)
    tf.summary.scalar("nonzero_fraction", nonzero_fraction)
    nonzero_hook = tf.train.LoggingTensorHook(
      tensors={
        "predicted_nonzero_fraction": nonzero_fraction,
        "true_nonzero_fraction": true_nonzero_fraction,
      },
      every_n_iter=50,
    )
    hooks.append(nonzero_hook)


    def _class_to_image(category):
      return tf.cast(category, tf.float32)[..., tf.newaxis]
    dist_image = _class_to_image(distribution_class)
    pred_image = _class_to_image(prediction_class)

    image_hook = tf.train.LoggingTensorHook(
      tensors={"distribution": dist_image[0, ..., 0],
               "prediction": pred_image[0, ..., 0],},
      every_n_iter=50,
    )
    hooks.append(image_hook)

    tf.summary.image("distributions", dist_image, 1)
    tf.summary.image("predictions", pred_image, 1)
    tf.summary.image("difference", (dist_image - pred_image) ** 2, 1)

    # Visualize output of predictions as categories.
    dist_summary = tf.summary.tensor_summary("distribution_tensor", dist_image)
    pred_summary = tf.summary.tensor_summary("predictions_tensor", pred_image)
    diff_summary = tf.summary.tensor_summary("difference_tensor", dist_image-pred_image)

    predict_output = {
        "predictions": pred_image
    }

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportion = (tf.reduce_sum(
        distributions_quantized,
        axis=[0, 1, 2],
        keepdims=True,
        ) + 2 ** params.bit_depth) / (tf.reduce_sum(distributions_quantized) + 2 ** params.bit_depth)
    inv_proportion = 1 / proportion

    loss = 0
    if params.last_loss_only:
      prediction_class = prediction_classes[-1]
      inv_estimate = inv_estimates[-1]
      ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)

      def bets_and_rewards_fn(params):
        if params.rewards == "1/n":
          rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
        elif params.rewards == "-logn":
          rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) * distributions_quantized,
            axis=-1)
        elif params.rewards == "log1/n":
          rewards = tf.reduce_sum(tf.math.log(inv_proportion) * distributions_quantized,
            axis=-1)
        elif params.rewards == "1/sqrtn":
          rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1))
        else:
          rewards = ones_like

        one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)

        if params.bets == "1/n":
          bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
        elif params.bets== "-logn":
          bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
            axis=-1)
        elif params.bets == "log1/n":
          bets = tf.reduce_sum(tf.math.log(inv_proportion) * one_hot_predictions,
            axis=-1)
        elif params.bets == "1/sqrtn":
          bets = tf.math.sqrt(tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1))
        else:
          bets = ones_like

        return bets, rewards


      bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
        lambda: (ones_like, ones_like))

      proportional_weights = bets * rewards
      if params.diff_scale == "abs":
        proportional_weights *= tf.cast(tf.math.abs(distribution_class - prediction_class), tf.float32)
      elif params.diff_scale == "square":
        proportional_weights *= tf.cast(tf.math.square(distribution_class - prediction_class), tf.float32)

      softmax_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=distributions_quantized,
        logits=inv_estimate,
        weights=proportional_weights
      )
      loss += softmax_loss

    else:
      for i, (prediction_class, inv_estimate) in enumerate(zip(prediction_classes, inv_estimates)):

        ones_like = tf.cast(tf.ones_like(prediction_class), tf.float32)

        def bets_and_rewards_fn(params):
          if params.rewards == "1/n":
            rewards = tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1)
          elif params.rewards == "-logn":
            rewards = - 1 * tf.reduce_sum(tf.math.log(proportion) * distributions_quantized,
              axis=-1)
          elif params.rewards == "log1/n":
            rewards = tf.reduce_sum(tf.math.log(inv_proportion) * distributions_quantized,
              axis=-1)
          elif params.rewards == "1/sqrtn":
            rewards = tf.math.sqrt(tf.reduce_sum(inv_proportion * distributions_quantized, axis=-1))
          else:
            rewards = ones_like

          one_hot_predictions = tf.one_hot(prediction_class, 2 ** params.bit_depth)

          if params.bets == "1/n":
            bets= tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1)
          elif params.bets== "-logn":
            bets = - 1 * tf.reduce_sum(tf.math.log(proportion) * one_hot_predictions,
              axis=-1)
          elif params.bets == "log1/n":
            bets = tf.reduce_sum(tf.math.log(inv_proportion) * one_hot_predictions,
              axis=-1)
          elif params.bets == "1/sqrtn":
            bets = tf.math.sqrt(tf.reduce_sum(inv_proportion * one_hot_predictions, axis=-1))
          else:
            bets = ones_like

          return bets, rewards


        bets, rewards = tf.cond(less_equal, lambda: bets_and_rewards_fn(params),
          lambda: (ones_like, ones_like))


        proportional_weights = bets * rewards
        if params.diff_scale == "abs":
          proportional_weights *= tf.cast(tf.math.abs(distribution_class - prediction_class), tf.float32)
        elif params.diff_scale == "square":
          proportional_weights *= tf.cast(tf.math.square(distribution_class - prediction_class), tf.float32)

        softmax_loss = tf.losses.softmax_cross_entropy(
          onehot_labels=distributions_quantized,
          logits=inv_estimate,
          weights=proportional_weights
        )

        tf.summary.scalar("prediction_loss", softmax_loss)

      loss += softmax_loss * (params.loss_scale ** i)

    if obs_estimates is not None:
      obs_loss = 0
      observation_truth = tf.split(observations, params.num_split, -1)
      shape = observations.shape
      observation_truth[0] = tf.zeros([tf.shape(observations)[0],
        int(shape[1]), int(shape[2]), int(shape[-1]) // params.num_split])
      for i, (est, obs) in enumerate(zip(obs_estimates, observation_truth)):
        obs_loss += tf.losses.mean_squared_error(obs, est) * (params.loss_scale ** i)
      tf.summary.scalar("observation_loss", obs_loss)
      loss += obs_loss

  with tf.variable_scope("optimizer"):
    learning_rate = tf.train.exponential_decay(
      learning_rate=lr,
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
    prediction_class = prediction_classes[-1]
    predictions_quantized = inv_estimates[-1]
    batch_accuracy = tf.reduce_mean(
      tf.cast(tf.equal(distribution_class, prediction_class), tf.float32))
    tf.summary.scalar("batch_accuracy", batch_accuracy)

    proportional_weights = loss_utils.inverse_class_weight(distributions_quantized)

    accuracy_hook = tf.train.LoggingTensorHook(
      tensors={"batch_accuracy": batch_accuracy,},
      every_n_iter=50
    )
    hooks.append(accuracy_hook)

    accuracy = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    accuracy_weighted = tf.metrics.accuracy(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1),
      weights=proportional_weights,
    )

    precision = tf.metrics.precision(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    recall = tf.metrics.recall(
      labels=tf.argmax(distributions_quantized, -1),
      predictions=tf.argmax(predictions_quantized, -1)
    )

    f1 = tf.where(tf.equal(precision[0] + recall[0], 0.),
      tf.constant(0, dtype=tf.float32), 2 * precision[0] * recall[0] / (precision[0] + recall[0]))

    labels = tf.argmax(distributions_quantized, -1)
    predictions= tf.argmax(predictions_quantized, -1)
    non_zero = tf.where(tf.equal(0, tf.cast(labels, dtype=tf.int32)), -1 * tf.ones_like(labels), labels)
    non_zero_correct = tf.math.reduce_sum(tf.cast(tf.equal(non_zero, predictions), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0, tf.cast(labels, dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0), tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

    tf.summary.scalar("non_zero_acc", non_zero_acc)
    tf.summary.scalar("precision", precision[0])
    tf.summary.scalar("recall", recall[0])
    tf.summary.scalar("f1", f1)


    eval_metric_ops = {
      "accuracy": accuracy,
      "accuracy_weighted": accuracy_weighted,
      "precision": precision,
      "recall": recall,
      "f1": tf.metrics.mean(f1),
      "non_zero_acc": tf.metrics.mean(non_zero_acc),
    }

    merged = tf.summary.merge([dist_summary, pred_summary, diff_summary])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    predictions=predict_output,
    eval_metric_ops=eval_metric_ops,
    training_hooks=hooks,
    evaluation_hooks=[tf.train.SummarySaverHook(save_steps=1, output_dir= params.job_dir + "/eval", summary_op = merged)]
  )


def input_fns_(
  example_shape: Tuple[int, int],
  observation_spec,
  frequency_indices,
  angle_indices,
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

  fns.append(preprocess.select_frequencies(frequency_indices))
  fns.append(preprocess.select_angles(angle_indices))

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
    frequency_indices=args.frequency_indices,
    angle_indices=args.angle_indices
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

  parser.add_argument(
    '--frequency_indices',
    type=lambda s: [int(index) for index in s.split(',')],
    required=True,
    default='0'
  )

  parser.add_argument(
    '--angle_indices',
    type=lambda s: [int(index) for index in s.split(',')],
    required=True,
    default='0'
  )

  parser.add_argument('--cloud_train', action='store_true')
  parser.set_defaults(cloud_train=False)

  args, _ = parser.parse_known_args()

  return args
