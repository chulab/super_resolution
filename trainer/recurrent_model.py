import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, ReLU, Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization, Input, Lambda, Dense
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
  )


class DownSample(Layer):

  def __init__(self, filters, kernel_size=[3, 3], strides=[2, 2]):
    super(DownSample, self).__init__()
    self.conv_1 = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal')
    self.conv_2 = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal')
    self.pool = MaxPool2D(strides)

  def call(self, inputs):
    x = self.conv_1(inputs)
    x = self.conv_2(x)
    pooled = self.pool(x)
    return x, pooled

class AddAttention(Layer):

  def __init__(self, intermediate_filters):
    super(AddAttention, self).__init__()
    self.proj_q = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal')
    self.proj_k = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal')
    self.relu = ReLU()
    self.post_conv = Conv2D(1, [1, 1], padding="same", kernel_initializer = 'he_normal')

  def call(self, inputs):
    [query, key] = inputs
    proj_q = self.proj_q(query)
    proj_k = self.proj_k(key)
    relued = self.relu(proj_q + proj_k)
    res = self.post_conv(relued)
    return tf.math.sigmoid(res)

class UpConcat(Layer):

  def __init__(self, filters, attention, up_kernel_size, kernel_size, name='upconcat'):
    super(UpConcat, self).__init__()
    with tf.name_scope(name):
      self.conv = Conv2D(filters, up_kernel_size, padding="same",
             kernel_initializer = 'he_normal')
      if attention:
        self.attention = AddAttention(32)
      else:
        self.attention = None

  def call(self, inputs):
    [input, skip_input] = inputs
    res = tf.image.resize_images(input, skip_input.shape[1:3],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    res = self.conv(res)
    if self.attention:
      skip_input = skip_input * self.attention([skip_input, res])
    res = tf.concat([res, skip_input], -1)
    return res

class Conv2DN(Layer):

  def __init__(self, filters):
    super(Conv2DN, self).__init__()
    self.conv = Conv2D(filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal')
    self.norm = BatchNormalization()

  def call(self, inputs):
    res = self.conv(inputs)
    res = self.norm(res)
    return res

class ResizeCombine(Layer):
  """
  Concatenation for Res-UNet
  """
  def __init__(self, filters):
    super(ResizeCombine, self).__init__()
    self.conv1 = Conv2DN(filters)
    self.conv2 = Conv2DN(filters)
    self.relu = ReLU()

  def call(self, inputs):
    input, skip_input = inputs
    res = tf.image.resize_images(input, skip_input.shape[1:3],
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
    res = self.conv1(res)
    res = self.relu(res)
    res = tf.concat([res, skip_input], -1)
    res = self.conv2(res)

    return res


class SqueezeExcitation(Layer):

  def __init__(self, out_dim ,ratio):
    super(SqueezeExcitation, self).__init__()
    self.out_dim = out_dim
    self.dense_1 = Dense(out_dim // ratio, activation='relu')
    self.dense_2 = Dense(out_dim, activation='sigmoid')


  def call(self, inputs):
    squeeze = global_avg_pool(inputs, name='global_avg_pooling')
    excitation = self.dense_1(squeeze)
    excitation = self.dense_2(excitation)
    excitation = tf.reshape(excitation, [-1,1,1,self.out_dim])
    scaled = inputs * excitation
    return scaled


class ResBlock(Layer):

  def __init__(self, se_block=False, kernel_size=[3, 3]):
    super(ResBlock, self).__init__()
    self.se_block = se_block
    self.kernel_size = kernel_size
    self.norm1 = BatchNormalization()
    self.norm2 = BatchNormalization()
    self.relu1 = ReLU()
    self.relu2 = ReLU()

  def build(self, input_shape):
    filters = int(input_shape[-1])
    self.conv1 = Conv2D(filters, self.kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal')
    self.conv2 = Conv2D(filters, self.kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal')
    if self.se_block:
      self.se = SqueezeExcitation(filters, 4)
    else:
      self.se = None
    super(ResBlock, self).build(input_shape)

  def call(self, inputs):
    res = self.norm1(inputs)
    res = self.relu1(res)
    res = self.conv1(res)
    res = self.norm2(res)
    res = self.relu2(res)
    res = self.conv2(res)
    if self.se:
      res = self.se(res)
    res += inputs
    return res

def down_sampling_block(input, filters, kernel_size=[3, 3], strides=[2, 2],
  se_block=False, name="down_sample"):
  with tf.name_scope(name):
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(input)
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = SqueezeExcitation(filters, 4)(res)
    pooled = MaxPool2D(strides).apply(res)
    return res, pooled


def res_block(input, se_block=False, kernel_size=[3, 3], name="res_block"):
  with tf.name_scope(name):
    filters = int(input.shape[-1])
    res = BatchNormalization().apply(input)
    res = ReLU().apply(res)
    res = Conv2D(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = BatchNormalization().apply(res)
    res = ReLU().apply(res)
    res = Conv2D(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = SqueezeExcitation(filters, 4)(res)
    res += input
    return res


def down_sampling_block_res(input, filters, se_block=False, kernel_size=[3, 3],
  strides=[2, 2], name="down_sample"):
  with tf.name_scope(name):
    res = ResBlock(se_block=se_block, kernel_size=kernel_size)(input)
    pooled = Conv2D(filters, [1, 1], activation=None, strides=strides,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    return res, pooled


def additive_attention(query, key, intermediate_filters, name="add_attn"):
  with tf.name_scope(name):
    proj_q = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(query)
    proj_k = Conv2D(intermediate_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(key)
    relued = ReLU().apply(proj_q + proj_k)
    res = Conv2D(1, [1, 1], padding="same", activation="sigmoid",
      kernel_initializer = 'he_normal').apply(relued)
    return res


def up_sampling_block(input, skip_input, filters, attention, up_kernel_size=[2,2],
  kernel_size=[3, 3], se_block=False, name="up_sample"):
  # filters is output filters
  with tf.name_scope(name):
    res = UpConcat(filters, attention, up_kernel_size, kernel_size).apply([input, skip_input])
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = SqueezeExcitation(filters, 4)(res)
    return res


def conv_2dn_block(input, filters, name='conv_2dn'):
  with tf.name_scope(name):
    res = Conv2D(filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(input)
    res = BatchNormalization().apply(res)
    return res


def combine_block(upsampled, skip_input, name='combine'):
  with tf.name_scope(name):
    res = ReLU().apply(upsampled)
    res = tf.concat([res, skip_input], -1)
    res = conv_2dn_block(res, skip_input.shape[-1])
    return res


def up_sampling_block_res(input, skip_input, filters, se_block=False,
  kernel_size=[3, 3], name="up_sample"):
  # ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data
  with tf.name_scope(name):
    # res = tf.image.resize_images(input, skip_input.shape[1:3],
    #   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
    # res = conv_2dn_block(res, filters)
    # combined = combine_block(res, skip_input)
    # return res_block(combined)
    res = ResizeCombine(filters)([input, skip_input])
    res = ResBlock(se_block=se_block, kernel_size=kernel_size)(res)
    return res


def conv_mid(input, filters, se_block=False, kernel_size=[3, 3], name="conv_mid"):
  with tf.name_scope(name):
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(input)
    res = Conv2D(filters, kernel_size, activation="relu",
      padding="same", kernel_initializer = 'he_normal').apply(res)
    if se_block:
      res = SqueezeExcitation(filters, 4)(res)
    return res


def conv_mid_res(input, se_block=False, name="conv_mid"):
  with tf.name_scope(name):
    res = ResBlock(se_block=se_block)(input)
    return res


def self_attention_embedding(input, filters, layers=3, name="attention_embed"):
  with tf.variable_scope(name):
    res = input
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
    res = cia.prepare_encoder(res, hparams, "local_2d")
    res = cia.transformer_encoder_layers(
        res,
        layers,
        hparams,
        attention_type=cia.AttentionType.LOCAL_2D,
        name="encoder")
    return res


def self_attention_mid(input, hparams, name="attention_mid"):
  # channels determined by hparams.hidden_size
  with tf.name_scope(name):
    res = cia.prepare_encoder(input, hparams, hparams.enc_attention_type)
    res = cia.transformer_encoder_layers(
        res,
        hparams.num_encoder_layers,
        hparams,
        attention_type=hparams.enc_attention_type,
        name="encoder")
    return res


def self_attention_net(input, filters, layers, output_filters, name="attention_net"):
  res = self_attention_embedding(input, filters, layers=layers)
  res = Conv2DN(output_filters)(res)
  return res


def unet(input, height, initial_filters, output_filters, dropout_rate=0.,
  name="unet", type="vanilla"):

  with tf.name_scope(name):
    pooled = input

    se_block = "se" in type
    skip_inputs = []
    for i in range(height):
      conved, pooled = down_sampling_block(pooled, initial_filters * (2 ** i)
        , se_block=se_block, name="down_sample_{}".format(i+1))
      skip_inputs.append(conved)

    up_sample_input = conv_mid(pooled, initial_filters * (2 ** height),
      se_block=se_block)

    res = up_sample_input

    attention = "attention" in type

    for i in range(1, height+1):
      res = up_sampling_block(res, skip_inputs[height-i],
        initial_filters * (2 ** (height-i)), attention,
        se_block=se_block, name="up_sample_{}".format(i))

    res = Conv2D(output_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(res)

    return Model(inputs=input, outputs=res)


def unet_res(input, height, initial_filters, output_filters, dropout_rate=0.,
  name="unet_res", type="residual"):
  with tf.name_scope(name):
    pooled = input

    se_block = "se" in type
    skip_inputs = []
    for i in range(1, height+1):
      conved, pooled = down_sampling_block_res(pooled, initial_filters * (2 ** i)
        , se_block=se_block, name="down_sample_{}".format(i))
      skip_inputs.append(conved)

    up_sample_input = conv_mid_res(pooled, se_block=se_block)
    res = up_sample_input

    for i in range(1, height+1):
      res = up_sampling_block_res(res, skip_inputs[height-i],
        initial_filters * (2 ** (height-i)), se_block=se_block,
        name="up_sample_{}".format(i))

    res = Conv2D(output_filters, [1, 1], padding="same",
      kernel_initializer = 'he_normal').apply(res)

    return Model(inputs=input, outputs=res)


def recurrent_unet(input, num_classes, forward_height, inverse_height,
  forward_filters, inverse_filters, forward_dropout, inverse_dropout,
  training, unet_type='residual', initial_inv_estimate=None, name="recurrent_unet"):
  with tf.name_scope(name):
    if unet_type == 'residual':
      unet_fn = unet_res
    else:
      unet_fn = unet

    if initial_inv_estimate is None:
      initial_inv_estimate = tf.zeros([tf.shape(input)[0], input.shape[1], input.shape[2]], dtype=tf.int32)
      initial_inv_estimate = tf.one_hot(initial_inv_estimate, num_classes)

    inv_estimates = []
    obs_estimates = []
    prev_inv_estimate = initial_inv_estimate
    for i, obs in enumerate(tf.split(input, input.shape[-1], -1)):
      embedded_inv = conv_2dn_block(prev_inv_estimate, forward_filters,
        name="inv_embedding_{}".format(i+1))
      obs_estimate = unet_fn(embedded_inv, forward_height, forward_filters,
        training, forward_dropout, name="forward_unet")
      with tf.variable_scope("forward_conv_{}".format(i+1)):
        obs_estimate = tf.keras.layers.Conv2D(
          filters=1,
          kernel_size=[1, 1],
          padding="same",
        ).apply(obs_estimate)
      obs_estimates.append(obs_estimate)
      concated = tf.concat([obs_estimate, obs], -1)
      embedded_concated = conv_2dn_block(concated, inverse_filters,
        name="obs_embedding_{}".format(i+1))
      inverse_shift = unet_fn(embedded_concated, inverse_height, inverse_filters,
        training, inverse_dropout, name="inverse_unet")
      with tf.variable_scope("inverse_conv_{}".format(i+1)):
        inverse_shift = tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=[1, 1],
          padding="same",
        ).apply(inverse_shift)
      prev_inv_estimate = inverse_shift + prev_inv_estimate
      inv_estimates.append(prev_inv_estimate)

    return obs_estimates, inv_estimates


def recurrent_unet_v2(input, num_classes, height, initial_filters, dropout,
  training, num_split, unet_type='attention_vanilla', initial_inv_estimate=None,
  name="recurrent_unet_v2"):

  assert(input.shape[-1] % num_split == 0), "Num_split must divide number of observations"

  with tf.name_scope(name):
    if unet_type == 'residual':
      unet_fn = unet_res
    elif unet_type == 'attention_vanilla':
      def unet_attention(input, height, initial_filters, training, dropout_rate, name):
        return unet(input, height, initial_filters, training, dropout_rate, name=name
          , attention=True)
      unet_fn = unet_attention
    else:
      unet_fn = unet

    split_size = int(input.shape[-1]) // num_split
    shape = input.shape
    concated_ex = Input([int(shape[1]), int(shape[2]), split_size + num_classes])
    inverse_unet = unet_fn(concated_ex, height, initial_filters,
      training, dropout, name="unet")

    if initial_inv_estimate is None:
      initial_inv_estimate = tf.zeros([tf.shape(input)[0], input.shape[1], input.shape[2]], dtype=tf.int32)
      initial_inv_estimate = tf.one_hot(initial_inv_estimate, num_classes)

    inv_estimates = []
    prev_inv_estimate = initial_inv_estimate
    for i, obs in enumerate(tf.split(input, num_split, -1)):
      concated = tf.concat([prev_inv_estimate, obs], -1)
      # embedded_concated = conv_2dn_block(concated, initial_filters,
      #   name="concated_embedding_{}".format(i+1))
      # shift = unet_fn(embedded_concated, height, initial_filters,
      #   training, dropout, name=unet_type)
      shift = inverse_unet(concated)
      with tf.variable_scope("shift_conv_{}".format(i+1)):
        shift = tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=[1, 1],
          padding="same",
        ).apply(shift)
      prev_inv_estimate += shift
      inv_estimates.append(prev_inv_estimate)

    return None, inv_estimates


class ObsAttention(Layer):

  def __init__(self, intermediate_filters, kernel_size=[5, 5]):
    super(ObsAttention, self).__init__()
    self.conv_1 = Conv2D(intermediate_filters, kernel_size, padding="same", activation="relu",
      kernel_initializer = 'he_normal')
    self.conv_2 = Conv2D(1, [1, 1], padding="same", kernel_initializer = 'he_normal')

  def call(self, inputs):
    [estimate, truth] = inputs
    proj = self.conv_1(estimate-truth)
    res = self.conv_2(proj)
    return tf.math.sigmoid(res)

def obs_attention(estimate, truth, intermediate_filters, kernel_size=[5, 5],
  name='obs_attention'):
  with tf.name_scope(name):
    proj = Conv2D(intermediate_filters, kernel_size, padding="same", activation="relu",
      kernel_initializer = 'he_normal').apply(estimate-truth)
    res = Conv2D(1, [1, 1], padding="same", kernel_initializer = 'he_normal').apply(proj)
    coefficients = tf.math.sigmoid(res)
    return coefficients


def recurrent_unet_v3(input, num_classes, forward_height, inverse_height,
  forward_filters, inverse_filters, forward_dropout, inverse_dropout,
  training, num_split, unet_type='residual', initial_inv_estimate=None, name="recurrent_unet"):

  assert(input.shape[-1] % num_split == 0), "Num_split must divide number of observations"

  with tf.name_scope(name):
    if unet_type == 'residual':
      unet_fn = unet_res
    elif unet_type == 'attention_vanilla':
      def unet_attention(input, height, initial_filters, training, dropout_rate, name):
        return unet(input, height, initial_filters, training, dropout_rate, name=name
          , attention=True)
      unet_fn = unet_attention
    else:
      unet_fn = unet

    if initial_inv_estimate is None:
      initial_inv_estimate = tf.zeros([tf.shape(input)[0], input.shape[1], input.shape[2]], dtype=tf.int32)
      initial_inv_estimate = tf.one_hot(initial_inv_estimate, num_classes)

    split_size = int(input.shape[-1]) // num_split
    inv_estimates = []
    obs_estimates = []
    prev_inv_estimate = initial_inv_estimate
    for i, obs in enumerate(tf.split(input, num_split, -1)):
      embedded_inv = conv_2dn_block(prev_inv_estimate, forward_filters,
        name="inv_embedding_{}".format(i+1))
      obs_estimate = unet_fn(embedded_inv, forward_height, forward_filters,
        training, forward_dropout, name="forward_unet")
      with tf.variable_scope("forward_conv_{}".format(i+1)):
        obs_estimate = tf.keras.layers.Conv2D(
          filters=split_size,
          kernel_size=[1, 1],
          padding="same",
        ).apply(obs_estimate)
      obs_estimates.append(obs_estimate)

      forget_coefficients = obs_attention(obs_estimate, obs, 32, name="forget_gate")
      input_coefficients = obs_attention(obs_estimate, obs, 32, name="input_gate")

      concated = tf.concat([obs_estimate, obs], -1)
      embedded_concated = conv_2dn_block(concated, inverse_filters,
        name="obs_embedding_{}".format(i+1))
      inverse_shift = unet_fn(embedded_concated, inverse_height, inverse_filters,
        training, inverse_dropout, name="inverse_unet")
      with tf.variable_scope("inverse_conv_{}".format(i+1)):
        inverse_shift = tf.keras.layers.Conv2D(
          filters=num_classes,
          kernel_size=[1, 1],
          padding="same",
        ).apply(inverse_shift)
      prev_inv_estimate = input_coefficients * inverse_shift + forget_coefficients * prev_inv_estimate

      # prev_inv_estimate = inverse_shift + prev_inv_estimate
      inv_estimates.append(prev_inv_estimate)

    return obs_estimates, inv_estimates

def get_embedding(input, embedding_string, filters, name="embedding"):
  with tf.name_scope(name):
    res = input
    for block in embedding_string.split('_'):
      if block == 'conv':
        res = Conv2D(filters, [1, 1], padding="same", kernel_initializer = 'he_normal').apply(res)
      elif block == 'attention':
        res = self_attention_embedding(res, int(res.shape[-1]), name=name+'_attention')
      elif block == 'se':
        res = SqueezeExcitation(int(res.shape[-1]), 4)(res)
    return res


def recurrent_unet_v4(input, num_classes, forward_height, inverse_height,
  forward_filters, inverse_filters, forward_dropout, inverse_dropout,
  training, num_split, unet_type='residual', initial_inv_hidden=None,
  initial_inv_cell=None, gating='LSTM', embedding='conv_attention',
  name="recurrent_unet"):
  '''
  LSTM Architecture
  '''

  assert(input.shape[-1] % num_split == 0), "Num_split must divide number of observations"

  with tf.name_scope(name):
    if 'res' in unet_type:
      unet_fn = unet_res
    else:
      unet_fn = unet

    one_hot_zeros = tf.zeros([tf.shape(input)[0], input.shape[1], input.shape[2]], dtype=tf.int32)
    one_hot_zeros = tf.one_hot(one_hot_zeros, num_classes)
    if initial_inv_hidden is None:
      initial_inv_estimate = one_hot_zeros
    if initial_inv_cell is None:
      initial_inv_cell = one_hot_zeros

    split_size = int(input.shape[-1]) // num_split
    inv_estimates = []
    obs_estimates = []
    prev_inv_estimate = initial_inv_estimate
    prev_inv_cell = initial_inv_cell

    shape = input.shape
    embedded_inv_ex = Input([int(shape[1]), int(shape[2]),
      forward_filters])
    embedded_concated_ex = Input([int(shape[1]), int(shape[2]),
      inverse_filters])
    forward_unet = unet_fn(embedded_inv_ex, forward_height, forward_filters,
      split_size, forward_dropout, type=unet_type, name="forward_unet")
    inverse_unet = unet_fn(embedded_concated_ex, inverse_height, inverse_filters,
      num_classes, inverse_dropout, type=unet_type, name="inverse_unet")

    if gating == 'LSTM':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)
      output_gate = ObsAttention(32)
    elif gating == 'GRU':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)

    for i, obs in enumerate(tf.split(input, num_split, -1)):
      embedded_inv = get_embedding(prev_inv_estimate, embedding,
        forward_filters, name="inv_embedding_{}".format(i+1))
      obs_estimate = forward_unet(embedded_inv)
      obs_estimates.append(obs_estimate)

      if gating == 'LSTM':
        forget_coefficients = forget_gate([obs_estimate, obs])
        input_coefficients = input_gate([obs_estimate, obs])
        output_coefficients = output_gate([obs_estimate, obs])
      elif gating == 'GRU':
        forget_coefficients = forget_gate([obs_estimate, obs])
        input_coefficients = input_gate([obs_estimate, obs])
        output_coefficients =  1
      else:
        forget_coefficients = input_coefficients = output_coefficients =  1

      concated = tf.concat([obs_estimate, obs], -1)
      embedded_concated = get_embedding(concated, embedding,
        inverse_filters, name="obs_embedding_{}".format(i+1))
      inverse_shift = inverse_unet(embedded_concated)

      prev_inv_cell = input_coefficients * inverse_shift + forget_coefficients * prev_inv_cell
      prev_inv_estimate = output_coefficients * prev_inv_cell

      inv_estimates.append(prev_inv_estimate)

    return obs_estimates, inv_estimates

def bidirectional_recurrent_unet_v4(input, num_classes, forward_height, inverse_height,
  forward_filters, inverse_filters, forward_dropout, inverse_dropout,
  training, num_split, unet_type='residual', initial_inv_hidden=None,
  initial_inv_cell=None, name="bidirectional_recurrent_unet"):

  with tf.name_scope(name):
    forward_obs_estimates, forward_inv_estimates = recurrent_unet_v4(
      input, num_classes, forward_height, inverse_height, forward_filters,
      inverse_filters, forward_dropout, inverse_dropout, training, num_split,
      unet_type, initial_inv_hidden, initial_inv_cell, name="forward_rnet")
    reverse_input = tf.reverse(input, [-1])
    reverse_obs_estimates, reverse_inv_estimates = recurrent_unet_v4(
      reverse_input, num_classes, forward_height, inverse_height, forward_filters,
      inverse_filters, forward_dropout, inverse_dropout, training, num_split,
      unet_type, initial_inv_hidden, initial_inv_cell, name="forward_rnet")
    reverse_obs_estimates.reverse()
    reverse_inv_estimates.reverse()

    final_obs_estimates = [(obs_1 + obs_2) / 2 for (obs_1, obs_2) in
      zip(forward_obs_estimates, reverse_obs_estimates)]
    final_inv_estimates = [(inv_1 + inv_2) / 2 for (inv_1, inv_2) in
      zip(forward_inv_estimates, reverse_inv_estimates)]

    return final_obs_estimates, reverse_inv_estimates


def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def squeeze_excitation_block(input, out_dim, ratio, layer_name="se_block"):
    with tf.name_scope(layer_name) :

        squeeze = global_avg_pool(input, name='global_avg_pooling')

        excitation = Dense(out_dim // ratio, activation='relu').apply(squeeze)
        excitation = Dense(out_dim, activation='sigmoid').apply(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scaled = input * excitation

        return scaled

def se_res_block(input, kernel_size=[3, 3], reduction_ratio=4, name="se_res_block"):
  with tf.name_scope(name):
    filters = int(input.shape[-1])
    res = BatchNormalization().apply(input)
    res = ReLU().apply(res)
    res = Conv2D(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = BatchNormalization().apply(res)
    res = ReLU().apply(res)
    res = Conv2D(filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = squeeze_excitation_block(res, filters, reduction_ratio)
    res += input
    return res

def se_res_net(input, num_classes, num_blocks, name="se_res_net"):
  with tf.name_scope(name):
    res = input
    for i in range(1, num_blocks+1):
      res = se_res_block(input, name="se_res_block_{}".format(i))
    res = Conv2D(num_classes, [1, 1], padding="same", kernel_initializer = 'he_normal').apply(res)
    return res

class DenseNetBlock(Layer):

  def __init__(self, filters, num_blocks=5, kernel_size=[3, 3]):
    super(DenseNetBlock, self).__init__()
    tf.logging.info("Using Dense Net")
    self.convs = []
    for _ in range(num_blocks):
      self.convs.append(
        Conv2D(filters, kernel_size, padding="same", kernel_initializer = 'he_normal')
      )

  def call(self, inputs):
    storage = [inputs]
    for conv in self.convs:
      if len(storage) > 1:
        concated = tf.concat(storage, -1)
      else:
        concated = storage[0]
      storage.append(conv(concated))
    return storage[-1]



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

  # observations = tf.signal.fft2d(tf.cast(observations, dtype=tf.complex64))
  # observations = tf.concat([tf.real(observations), tf.imag(observations)], axis=-1)

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
  elif params.recurrent == "recurrent_v2":
    obs_estimates, inv_estimates = recurrent_unet_v2(observations, 2 ** params.bit_depth,
      params.unet_height, params.initial_filters, params.dropout, training, params.num_split,
      unet_type=params.unet_type)
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
