"""Contains basic modules for CNN models."""

import tensorflow as tf


def residual_block(inputs, channels, kernel_size, residual_scale):
  with tf.name_scope("residual_block"):
    res = tf.keras.layers.SeparableConv2D(
      channels,
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(inputs)
    res = tf.keras.layers.ReLU().apply(res)
    res = tf.keras.layers.SeparableConv2D(
      channels,
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(res)
    res = tf.keras.layers.Lambda(lambda x: x * residual_scale).apply(res)
    return tf.keras.layers.Add().apply([inputs, res])

def depthwise_residual_block(inputs, kernel_size, residual_scale):
  with tf.name_scope("residual_block"):
    res = tf.keras.layers.DepthwiseConv2D(
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(inputs)
    res = tf.keras.layers.ReLU().apply(res)
    res = tf.keras.layers.DepthwiseConv2D(
      kernel_size,
      activation=None,
      use_bias=False,
      padding="same"
    ).apply(res)
    res = tf.keras.layers.Lambda(lambda x: x * residual_scale).apply(res)
    return tf.keras.layers.Add().apply([inputs, res])

def spatial_block(x, scales, filters_per_scale, kernel_size,
                  activation=tf.nn.leaky_relu, use_bias=True):
  with tf.name_scope("spatial_block"):
    convs = []
    for scale in scales:
      conv = tf.keras.layers.SeparableConv2D(
        filters=filters_per_scale,
        kernel_size=kernel_size,
        dilation_rate=(scale, scale),
        padding="same",
        use_bias=use_bias,
        activation=activation,
        ).apply(x)
      conv = tf.keras.layers.BatchNormalization().apply(conv)
      convs.append(conv)
    if len(scales) > 1:
      net = tf.keras.layers.Concatenate().apply(convs)
    else:
      net = convs[0]
    return net


def downsample_block(x, depth_multiplier=2, kernel_size=5, stride=2):
  with tf.name_scope('downsample_block'):
    return tf.keras.layers.DepthwiseConv2D(
      kernel_size,
      strides=(stride, stride),
      padding="same",
      use_bias=False,
      activation=None,
      depth_multiplier=depth_multiplier,
    ).apply(x)


def donwnsampleModule(input_shape, downsample_factor, **kwargs):
  """Defines downsampling module."""
  with tf.name_scope("downsample_module"):
    inputs = tf.keras.Input(shape=input_shape)

    net = inputs

    for _ in range(downsample_factor):
      net = downsample_block(net, **kwargs)

    return tf.keras.Model(inputs=inputs, outputs=net)


def upsampleBlock(x, filters=64, kernel_size=5, stride=2):
  """Upsamples spatial dimension of x by `stride`."""
  return tf.keras.layers.Conv2DTranspose(
    filters=filters,
    kernel_size=kernel_size,
    strides=(stride, stride),
    padding="same",
    use_bias=False,
    activation=None,
  ).apply(x)
