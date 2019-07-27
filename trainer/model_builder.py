"""
Framework to build general models. There are generally three types of building
blocks:

  1. Embedding: used to preprocess input.
  2. Propagator: produces an output that is the same shape as input.
  3. Pooler: used to downsample input.

Different implementations of each building block are given. With these building
blocks, two types of models can be created according to `model_type` in HParams.

  1. `prop`: model is created from embedding, propagator and pooler in a linear
    fashion. Architecture is embedding -> (propagator + pooler) x n.

    Relevant parameters are:
      `embedding`: String of embedder names separated by `/`, `_{i}` can be
        added behind an embedder name to repeat the embedder {i} times. If the
        number of filters in an embedder's output is supported, one can add
        '-{filters}' behind an embedder name and infront of `_{i}` to specify
        output filters.

        Example: embedding='xception-128_2/attention' which applies xception
          entry flow twice with 128 output filters, before passing through a
          self-attention layer.

      `prop_layers`: List of ints denoting number of times to apply propagator
        in each block. Between blocks (i.e. between consecutive ints), the
        pooler is applied to downsample the input.
      `forward_prop`: Name of propagator to use. Must be in PROPAGATORS.
      `forward_kwargs`: kwargs to pass to propagator.
      `forward_conv_name`: Name of convolution to apply in pooling
        (if applicable).
      `pooler`: Name of pooler to use. Must be in POOLERS.
      `pooler_filters`: List of len(prop_layers) - 1 ints denoting number of
        filters after each pooling stage. If None (default), the number of
        filters is doubled after each pooling step.
      `dropout`: dropout rate.

  2. `unet`: model is created from embedding, propagator and pooler in a UNet
    fashion. Architecture is embedding -> UNet where UNet blocks are built from
    propagator and pooler.

    See `U-Net: Convolutional Networks for Biomedical Image Segmentation,
      Olaf Ronneberger, Philipp Fischer, Thomas Brox.`

    Relevant parameters are:
      `embedding`: See above.
      `forward_prop`: Name of propagator to use in downsampling section of UNET.
        Must be in PROPAGATORS.
      `forward_kwargs`: kwargs to pass to forward_prop.
      `forward_conv_name`: Name of convolution to apply in downsampling pooling
        (if applicable).
      `mid_prop`: Name of propagator to use in bottleneck section of UNET.
        Must be in PROPAGATORS.
      `mid_kwargs`: kwargs to pass to mid_prop.
      `reverse_prop`: Name of propagator to use in upsampling section of UNET.
        Must be in PROPAGATORS.
      `reverse_kwargs`: kwargs to pass to forward_prop.
      `reverse_conv_name`: Name of convolution to apply in upsampling pooling
        (if applicable).
      `pooler`: Name of pooler to use. Must be in POOLERS.
      `forward_height`: int representing length of downsampling branch.
      `reverse_height`: int representing length of upsampling branch.
      `dropout`: dropout rate.
      `unet_attention`: Bool indicating whether attention between upsampled
        image in the upsampling branch of UNet and the skip-connection from the
        corresponding image along the downsampling branch should be applied.

        See `Attention U-Net: Learning Where to Look for the Pancreas.`


Recurrent architectures can also be built from either of the above models by
specifying the `recurrent` parameter. In general, the philosophy of a recurrent
model is to split the different observations (channels) into groups of size
`channels_per_step`. At each timestep, a group of observations is fed in and a
new prediction is made based on these observations and the prediction from the
previous timestep. Two types of recurrent architectures are supported.

  1. `recurrent_v1`: Learns both forward and inverse processes concurrently.
    In the following description, observations are obtained from applying the
    forward process to sources (e.g. scatterers). Models are built according
    to the specifications described above (either `prop` or `unet`).

    Architecture (after the embedding layer) is of the form:
      Previous source estimate -> Forward model -> New obs estimate.
      New obs estimate, True obs -> Inverse model -> New source estimate.

    Then, loss functions can be applied on obs and source estimates at each step
    to train both forward and inverse models.

  2. `recurrent_v2`: Only learns the inverse process.

    Architecture (after the embedding layer) is of the form:
      Previous source estimate, True obs -> Model -> New source estimate.

    A loss function is then applied on the source estimates at each step.

  In addition to `prop` or `unet` parameters described above, a recurrent model
  uses:

    `gating`: string which is one of None, `LSTM` or `GRU`.
    `channels_per_step`: Number of observations in one group.
"""


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
    recurrent='None',
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
    embedding="xception/attention",
    gating="LSTM",
    model_type="prop",
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
    pooler_filters=None,
    unet_attention=True,
    channels_per_step=2,
  )

CONV_FNS = ["conv", "depthconv", "sepconv", "conv2dn"]
PROPAGATORS = ["vanilla", "residual", "attention", "dense", "xception"]
POOLERS = ['maxpool', 'depthwise_conv', 'conv', "residual", "xception"]

class Conv2DN(Layer):
  """
  Conv2D with batch normalization, usually used with 1x1 kernel to change
  filters.
  """
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
  """
  Computes additive attention coefficients given query and key.

  Arguments:
    query: `tf.Tensor` with shape [B, H, W, C].
    key: `tf.Tensor` with shape [B, H, W, C'].
    intermediate_filters: int denoting channel dimension to project
      query and key.

  Returns:
    `tf.Tensor` with shape [B, H, W] representing attention coefficients.
  """

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
  """
  Modifies input along channels based on the squeeze-excitation procedure.

  Arguments:
    input: `tf.Tensor` with shape [B, H, W, C].
    ratio: int denoting intermediate channels to use for computation. ratio must
      divide C above.

  Returns:
    input: `tf.Tensor` with shape [B, H, W, C].
  """
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
  """
  Attention based on difference between an estimate and ground truth.
  """
  def __init__(self, intermediate_filters, kernel_size=[5, 5]):
    super(ObsAttention, self).__init__()
    self.conv_1 = Conv2D(intermediate_filters, kernel_size, padding="same",
      activation="relu", kernel_initializer = 'he_normal')
    self.conv_2 = Conv2D(1, [1, 1], padding="same", activation='sigmoid')

  def call(self, inputs):
    [estimate, truth] = inputs
    proj = self.conv_1(estimate-truth)
    res = self.conv_2(proj)
    return res

class AddAttention(Layer):
  """
  tf.keras.Layer implemention of additive_attention for purposes of reusing.
  """
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
  """Two convolutions without BN."""
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
  """ResNet block."""
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
  """Self-attention block."""
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
          name=name+"_encoder")
      return res

    res = Lambda(attention_).apply(input)
    return res


def dense_propagator(input, conv_name, layers=3, kernel_size=[3, 3],
  name="dense_prop"):
  """DenseNet block."""
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    filters = int(input.shape[-1])
    concated = input
    for _ in range(layers):
      conved = conv_fn(filters, kernel_size, padding="same",
        kernel_initializer='he_normal', activation='relu').apply(concated)
      concated = Concatenate().apply([concated, conved])

    return conved


def xception_propagator(input, conv_name, se_block=False, kernel_size=[3,3],
  name="xception_prop"):
  """Xception block."""
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    filters = int(input.shape[-1])

    res = ReLU().apply(input)
    res = conv_fn(filters=filters, kernel_size=kernel_size,
      padding="same").apply(res)
    res = BatchNormalization().apply(res)
    res = ReLU().apply(res)
    res = conv_fn(filters=filters, kernel_size=kernel_size,
      padding="same", dilation_rate=2).apply(res)
    res = BatchNormalization().apply(res)
    res = ReLU().apply(res)
    res = conv_fn(filters=filters, kernel_size=kernel_size,
      padding="same", dilation_rate=4).apply(res)
    res = BatchNormalization().apply(res)
    if se_block:
      res = squeeze_excitation(res)

    res = Add().apply([res, input])
    return res


def get_propagator(name):
  assert name in PROPAGATORS, "Propagator must be in {}".format(PROPAGATORS)

  if name == "vanilla":
    return vanilla_propagator
  elif name == "residual":
    return res_propagator
  elif name == "attention":
    return attention_propagator
  elif name == 'xception':
    return xception_propagator
  else:
    return dense_propagator


def get_embedding(input, embedding_string, name="embedding"):
  with tf.name_scope(name):
    res = input
    for j, block in enumerate(embedding_string.split('/')):
      if '_' in block:
        fn_name = block.split('_')[0]
        number = int(block.split('_')[1])
      else:
        fn_name = block
        number = 1

      if 'conv' in fn_name:
        splitted = fn_name.split('-')
        fn_name = splitted[0]
        filters = int(splitted[1])
        conv_fn = get_conv_fn(fn_name)
        for _ in range(number):
          res = conv_fn(filters, [1, 1], padding="same",
            kernel_initializer = 'he_normal').apply(res)
      elif fn_name == 'attention':
        for i in range(1, number+1):
          res = attention_propagator(res,
            name=name+'_attention_{}_{}'.format(j, i))
      elif fn_name == 'se':
        for _ in range(number):
          res = squeeze_excitation(res)
      elif 'xception' in fn_name or 'residual' in fn_name:
        if 'xception' in fn_name:
          conv_name = 'sepconv'
        else:
          conv_name = 'conv'

        desired_filters = None
        if '-' in fn_name:
          splitted = fn_name.split('-')
          fn_name = splitted[0]
          desired_filters = int(splitted[1])

        for _ in range(number):
          if desired_filters is None:
            pooled_filters = int(res.shape[-1]) * 2
          else:
            pooled_filters = desired_filters
          res = perform_pooling(res, pooled_filters, fn_name,
            conv_name=conv_name)
    return res

def build_model_from_propagator(input, hparams, num_classes=None,
  name='propagator_model'):
  """
  Builds `prop` model.

  Args:
    input: tf.keras.Input with shape of real input.
    hparams: model's hyperparameters.
    num_classes: if given, post 1x1 Conv2D is performed to reduced filters to
      num_classes to represent class logits.
    name: namescope.
  """
  with tf.name_scope(name):
    prop_fn = get_propagator(hparams.forward_prop)

    res = input
    for i, elem in enumerate(hparams.prop_layers):
      for j in range(1, elem+1):
        res = prop_fn(res, name=hparams.forward_prop + '_{}_{}'.format(i+1, j),
          **hparams.forward_kwargs)
      if i < len(hparams.prop_layers) - 1:
        if hparams.pooler_filters is not None:
          pooled_filters = hparams.pooler_filters[i]
        else:
          pooled_filters = int(res.shape[-1]) * 2
        res = perform_pooling(res, pooled_filters, hparams.pooler,
          conv_name=hparams.forward_conv_name, name='pooling_{}'.format(i+1))


    res = Dropout(rate=hparams.dropout).apply(res)
    if num_classes is not None:
      res = Conv2D(num_classes, [1, 1], padding="same",
        kernel_initializer = 'he_normal', use_bias=False).apply(res)

    return Model(inputs=input, outputs=res)


def xception_pooler(input, pooled_filters, conv_name='sepconv',
  kernel_size=[3, 3], name="xception_pooler"):
  """Xception entry flow block."""
  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    res = conv_fn(pooled_filters, kernel_size, activation='relu',
      padding="same", kernel_initializer = 'he_normal').apply(input)
    res = BatchNormalization().apply(res)
    res = conv_fn(pooled_filters, kernel_size, activation=None,
      padding="same", kernel_initializer = 'he_normal').apply(res)
    res = BatchNormalization().apply(res)
    res = MaxPool2D(pool_size=(2, 2)).apply(res)
    projected = Conv2D(pooled_filters, kernel_size=[1, 1], strides=[2, 2],
      padding="same").apply(input)
    projected = BatchNormalization().apply(projected)
    return Add().apply([res, projected])


def resnet_pooler(input, pooled_filters, conv_name='conv', kernel_size=[3, 3]
  , name="resnet_pooler"):
  """ResNet pooling block."""
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
  """Downsampling block of UNet.

  Args:
    input: tf.Tensor with shape [B, H, W, C].
    pooled_filters: number of filters in pooled output.
    propagator: propagator_fn.
    prop_kwargs: kwargs for propagator.
    conv_name: name of conv_fn for pooling.
    kernel_size: kernel for pooling.
    activation: activation for pooling.
    pooler: name of pooler. Must be in POOLERS.
    name: namescope.

  Returns:
    res: result of propagator applied to input (to be used for skip-connection)
      with same shape as input [B, H, W, C].
    pooled: res after pooling, with shape [B, H/2, W/2, pooled_filters].
  """
  with tf.name_scope(name):
    res = propagator(input, **prop_kwargs)
    pooled = perform_pooling(res, pooled_filters, pooler, conv_name=conv_name,
      kernel_size=kernel_size)

    return res, pooled


def conv_concat(input, skip_input, filters, conv_fn=Conv2D,
  up_kernel_size=[2, 2], kernel_size=[3, 3], attention=True, name='conv_concat'):
  """Upsamples input along upsampling UNet branch and concatenates with
  corresponding skip_input from downsampling branch.

  Args:
    input: tf.Tensor with shape [B, H, W, C].
    skip_input: tf.Tensor from corresponding downsampling_branch with shape
      [B, H', W', C].
    filters: number of filters of concatenated output.
    conv_fn: conv_fn to use after upsampling and concatenation.
    up_kernel_size: kernel after upsampling.
    kernel_size: kernel after concatenation.
    attention: Bool indicating whether additive attention between input and
      skip_input should be applied to modify skip_input.
    name: namescope.

  Returns:
    res: tf.Tensor with shape [B, H', W', filters].
  """
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
  """Upsampling block of UNet.

  Args:
    input: tf.Tensor with shape [B, H, W, C].
    skip_input: tf.Tensor from corresponding downsampling_branch with shape
      [B, H', W', C].
    filters: number of filters of output.
    propagator: prop_fn to use.
    prop_kwargs kwargs for propagator.
    conv_name: name of conv_fn to use in conv_concat.
    up_kernel_size: kernel after upsampling in conv_concat
    kernel_size: kernel after concatenation in conv_concat.
    attention: Bool indicating whether additive attention between input and
      skip_input should be applied to modify skip_input.
    name: namescope.

  Returns:
    res: tf.Tensor with shape [B, H', W', filters].
  """

  with tf.name_scope(name):
    conv_fn = get_conv_fn(conv_name)
    res = conv_concat(input, skip_input, filters, conv_fn, up_kernel_size,
      kernel_size, attention)
    res = propagator(res, **prop_kwargs)
    return res


def build_unet_from_propagator(input, hparams, num_classes=None, name='unet'):
  """
  Builds `unet` model.

  Args:
    input: tf.keras.Input with shape of real input.
    hparams: model's hyperparameters.
    num_classes: if given, post 1x1 Conv2D is performed to reduced filters to
      num_classes to represent class logits.
    name: namescope.
  """
  with tf.name_scope(name):
    pooled = input
    skip_inputs = []
    forward_prop_fn = get_propagator(hparams.forward_prop)
    initial_filters = int(input.shape[-1])

    # downsampling branch
    for i in range(1, hparams.forward_height+1):
      conved, pooled = down_sampling_block(pooled, initial_filters * (2 ** i),
        forward_prop_fn, hparams.forward_kwargs,
        conv_name=hparams.forward_conv_name, pooler=hparams.pooler,
        name="down_sample_{}".format(i))
      if i > hparams.forward_height - hparams.reverse_height:
        skip_inputs.append(conved)

    # middle/bottleneck branch
    mid_prop_fn = get_propagator(hparams.mid_prop)
    res = mid_prop_fn(pooled, name='mid_prop', **hparams.mid_kwargs)

    # upsampling branch
    reverse_prop_fn = get_propagator(hparams.reverse_prop)
    for i in range(1, hparams.reverse_height+1):
      res = up_sampling_block(res, skip_inputs[hparams.reverse_height - i],
        initial_filters * (2 ** (hparams.forward_height-i)),
        reverse_prop_fn, hparams.reverse_kwargs,
        conv_name=hparams.reverse_conv_name,
        attention=hparams.unet_attention, name="up_sample_{}".format(i))

    res = Dropout(rate=hparams.dropout).apply(res)
    if num_classes is not None:
      res = Conv2D(num_classes, [1, 1], padding="same",
        kernel_initializer = 'he_normal', use_bias=False).apply(res)

    return Model(inputs=input, outputs=res)


def recurrent_net_v1(input, num_classes, hparams, initial_inv_hidden=None,
  initial_inv_cell=None, name="recurrent_unet_v1"):
  """
  Recurrent net that learns both forward and inverse processes.

  Args:
    input: tf.Tensor with shape [B, H, W, C].
    num_classes: number of class labels.
    hparams: model's hyperparameters.
    initial_inv_hidden: tf.Tensor with shape [B, H, W, num_classes] that is
      initial hidden state of network.
    initial_inv_cell: tf.Tensor with shape [B, H, W, num_classes] that is
      initial cell state of network.
    name: namescope.

  Returns:
    List of C / hparams.channels_per_step observation estimates, each being a
      tf.Tensor with shape [B, H, W, hparams.channels_per_step].
    List of C / hparams.channels_per_step inverse estimates, each being a
      tf.Tensor with shape [B, H, W, num_classes].
  """
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
      prev_inv_estimate = one_hot_zeros
    else:
      prev_inv_estimate = initial_inv_hidden

    if initial_inv_cell is None:
      prev_inv_cell = one_hot_zeros
    else:
      prev_inv_cell = initial_inv_cell


    split_size = hparams.channels_per_step
    num_split = int(input.shape[-1]) // split_size
    inv_estimates = []
    obs_estimates = []

    inv_embedded_ex = get_embedding(prev_inv_estimate, hparams.embedding,
      name='inv_ex')
    inv_example = Input(tensor=inv_embedded_ex)
    obs_ex = tf.split(input, num_split, -1)[0]
    concated_ex = tf.tile(obs_ex, [1, 1, 1, 2])
    concated_embedded_ex = get_embedding(concated_ex, hparams.embedding,
      name='concated_ex')
    for_example = Input(tensor=concated_embedded_ex)

    if hparams.model_type == 'unet':
      forward_net = build_unet_from_propagator(inv_example, hparams,
        split_size, name="forward_unet")
      inverse_net = build_unet_from_propagator(for_example, hparams,
        num_classes, name="inverse_unet")
    else:
      forward_net = build_model_from_propagator(inv_example, hparams,
        split_size, name="forward_net")
      inverse_net = build_model_from_propagator(for_example, hparams,
        num_classes, name="inverse_net")

    if hparams.gating == 'LSTM':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)
      output_gate = ObsAttention(32)
    elif hparams.gating == 'GRU':
      forget_gate = ObsAttention(32)
      input_gate = ObsAttention(32)

    for i, obs in enumerate(tf.split(input, num_split, -1)):
      embedded_inv = get_embedding(prev_inv_estimate, hparams.embedding,
        name="inv_embedding_{}".format(i+1))
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
        name="obs_embedding_{}".format(i+1))
      inverse_shift = inverse_net(embedded_concated)

      prev_inv_cell = input_coefficients * inverse_shift + \
        forget_coefficients * prev_inv_cell
      prev_inv_estimate = output_coefficients * prev_inv_cell

      inv_estimates.append(prev_inv_estimate)

    return obs_estimates, inv_estimates


def recurrent_net_v2(input, num_classes, hparams, initial_inv_hidden=None,
  initial_inv_cell=None, name="recurrent_unet_v1"):
  """
  Recurrent net that learns both only inverse process.

  Args:
    input: tf.Tensor with shape [B, H, W, C].
    num_classes: number of class labels.
    hparams: model's hyperparameters.
    initial_inv_hidden: tf.Tensor with shape [B, H, W, num_classes] that is
      initial hidden state of network.
    initial_inv_cell: tf.Tensor with shape [B, H, W, num_classes] that is
      initial cell state of network.
    name: namescope.

  Returns:
    None (no observation estimates).
    List of C / hparams.channels_per_step inverse estimates, each being a
      tf.Tensor with shape [B, H, W, num_classes].
  """
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

    obs_ex = tf.split(input, num_split, -1)[0]
    concated_ex = tf.concat([prev_inv_estimate, obs_ex], -1)
    embedded_concated_ex = get_embedding(concated_ex, hparams.embedding,
      name="ex_embedding")
    example = Input(tensor=embedded_concated_ex)

    if hparams.model_type == 'unet':
      net = build_unet_from_propagator(example, hparams, num_classes,
        name="inverse_unet")
    else:
      net = build_model_from_propagator(example, hparams, num_classes,
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
        name="embedding_{}".format(i+1))
      shift = net(embedded_concated)
      prev_inv_cell = input_coefficients * shift + \
        forget_coefficients * prev_inv_cell
      prev_inv_estimate = output_coefficients * prev_inv_cell

      inv_estimates.append(prev_inv_estimate)

    return None, inv_estimates


def model_fn(features, labels, mode, params):
  """Defines model graph for non-recurrent model.

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
  distributions = labels

  tf.summary.image("original_distribution", distributions[..., tf.newaxis], 1)

  observations, distributions = preprocess.gpu_preprocess(observations,
    distributions, params)

  distributions_values, distributions_quantized = loss_utils.quantize_tensor(
    distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, ang_index in enumerate(params.angle_indices):
      angle = params.observation_spec.angles[ang_index]
      tf.summary.image("obs_angle_{}".format(angle), observations[..., i,
        tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)

  embedded = get_embedding(observations, params.embedding)
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
    distribution_class = loss_utils._logit_to_class(distributions_quantized)
    prediction_class = loss_utils._logit_to_class(predictions_quantized)

    # Visualize output of predictions as categories.
    tf.summary.tensor_summary("prediction_class", prediction_class)

    # Log fraction nonzero.
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class),
      tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class),
      tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(
      prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(
      prediction_class), tf.float32)
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
    diff_summary = tf.summary.tensor_summary("difference_tensor",
      dist_image-pred_image)

    predict_output = {
        "predictions": pred_image
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predict_output
    )

  # Loss. Compare output of nn to original images.
  with tf.variable_scope("loss"):
    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportional_weights = loss_utils.bets_and_rewards_weight(
      distributions_quantized, distribution_class, prediction_class, params)

    proportion_hook = tf.train.LoggingTensorHook(
      tensors={"proportional_weights": proportional_weights[0]},
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

    proportional_weights = loss_utils.inverse_class_weight(
      distributions_quantized)

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
      tf.constant(0, dtype=tf.float32), 2 * precision[0] * recall[0] /
      (precision[0] + recall[0]))

    labels = tf.argmax(distributions_quantized, -1)
    predictions= tf.argmax(predictions_quantized, -1)
    non_zero = tf.where(tf.equal(0, tf.cast(labels, dtype=tf.int32)),
      -1 * tf.ones_like(labels), labels)
    non_zero_correct = tf.math.reduce_sum(tf.cast(
      tf.equal(non_zero, predictions), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0,
      tf.cast(labels, dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0),
      tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

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
    evaluation_hooks=[tf.train.SummarySaverHook(
      save_steps=1,
      output_dir= params.job_dir + "/eval",
      summary_op = merged)]
  )


def model_fn_recurrent(features, labels, mode, params):
  """Defines model graph for recurrent model.

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

  observations, distributions = preprocess.gpu_preprocess(observations,
    distributions, params)

  logging.info("`observations` tensor after gpu preprocess in model is "
                "{}".format(observations))
  logging.info("`distributions` tensor  after gpu preprocess in model is "
                "{}".format(distributions))

  distributions_values, distributions_quantized = loss_utils.quantize_tensor(
    distributions, 2 ** params.bit_depth, 0., 2 ** params.bit_depth)

  # Average image along `channel` axis. This corresponds to previous SOA.
  averaged_observation = tf.reduce_mean(observations, axis=-1, keepdims=True)

  with tf.variable_scope("inputs"):
    # Add image summaries.
    for i, ang_index in enumerate(params.angle_indices):
      angle = params.observation_spec.angles[ang_index]
      tf.summary.image("obs_angle_{}".format(angle),
        observations[..., i, tf.newaxis], 1)
    tf.summary.image("averaged_observation", averaged_observation, 1)


  if params.recurrent == "recurrent_v1":
    obs_estimates, inv_estimates = recurrent_net_v1(observations,
      2 ** params.bit_depth, params)
  elif params.recurrent == "recurrent_v2":
    obs_estimates, inv_estimates = recurrent_net_v2(observations,
      2 ** params.bit_depth, params)

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
    predicted_nonzero_count = tf.cast(tf.count_nonzero(prediction_class),
      tf.float32)
    true_nonzero_count = tf.cast(tf.count_nonzero(distribution_class),
      tf.float32)
    true_nonzero_fraction = true_nonzero_count / tf.cast(tf.size(
      prediction_class), tf.float32)
    nonzero_fraction = predicted_nonzero_count / tf.cast(tf.size(
      prediction_class), tf.float32)
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
    diff_summary = tf.summary.tensor_summary("difference_tensor",
      dist_image-pred_image)

    predict_output = {
        "predictions": pred_image
    }

  # Loss. Compare output of nn to original images.
  loss = 0
  with tf.variable_scope("loss"):
    less_equal = tf.less_equal(tf.train.get_global_step(), params.scale_steps)
    lr = tf.cond(less_equal, lambda: tf.constant(params.learning_rate),
      lambda: tf.constant(params.learning_rate / 1000))

    proportion = (tf.reduce_sum(
        distributions_quantized,
        axis=[0, 1, 2],
        keepdims=True,
        ) + 2 ** params.bit_depth) / (tf.reduce_sum(distributions_quantized) + \
          2 ** params.bit_depth)
    inv_proportion = 1 / proportion

    if params.last_loss_only:
      prediction_class = prediction_classes[-1]
      proportional_weights = loss_utils.bets_and_rewards_weight(
        distributions_quantized, distribution_class, prediction_class, params)

      softmax_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=distributions_quantized,
        logits=inv_estimates[-1],
        weights=proportional_weights
      )
      loss += softmax_loss

    else:
      for i, (prediction_class, inv_estimate) in enumerate(zip(
        prediction_classes, inv_estimates)):

        proportional_weights = loss_utils.bets_and_rewards_weight(
          distributions_quantized, distribution_class, prediction_class, params)

        softmax_loss = tf.losses.softmax_cross_entropy(
          onehot_labels=distributions_quantized,
          logits=inv_estimate,
          weights=proportional_weights
        )

        loss += softmax_loss * (params.loss_scale ** i)

    if obs_estimates is not None:
      obs_loss = 0
      shape = observations.shape
      num_split = int(shape[-1]) // params.channels_per_step
      observation_truth = tf.split(observations, num_split, -1)
      observation_truth[0] = tf.zeros([tf.shape(observations)[0],
        int(shape[1]), int(shape[2]), params.channels_per_step])
      for i, (est, obs) in enumerate(zip(obs_estimates, observation_truth)):
        obs_loss += tf.losses.mean_squared_error(obs, est) * (params.loss_scale
          ** i)
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

    proportional_weights = loss_utils.inverse_class_weight(
      distributions_quantized)

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
      tf.constant(0, dtype=tf.float32),
      2 * precision[0] * recall[0] / (precision[0] + recall[0]))

    labels = tf.argmax(distributions_quantized, -1)
    predictions= tf.argmax(predictions_quantized, -1)
    non_zero = tf.where(tf.equal(0, tf.cast(labels, dtype=tf.int32)),
      -1 * tf.ones_like(labels), labels)
    non_zero_correct = tf.math.reduce_sum(tf.cast(tf.equal(non_zero,
      predictions), dtype=tf.int32))
    total_non_zero =tf.math.reduce_sum(tf.cast(tf.not_equal(0, tf.cast(labels,
      dtype=tf.int32)), dtype=tf.int32))
    non_zero_acc = tf.where(tf.equal(total_non_zero, 0),
      tf.constant(0, dtype=tf.float64), non_zero_correct / total_non_zero)

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
    evaluation_hooks=[tf.train.SummarySaverHook(
      save_steps=1,
      output_dir= params.job_dir + "/eval",
      summary_op = merged)]
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
    observation_shape=[len(observation_spec.angles)] + list(example_shape) + \
      [len(observation_spec.psf_descriptions)]))

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