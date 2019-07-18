import copy
import numpy as np
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.data_generators import image_utils


from tensor2tensor.data_generators import problem_hparams

import tensorflow as tf
import six
import collections

def custom_postprocess_image(x, rows, cols, hparams, target_vocab_size):
  """Postprocessing after decoding.

  Args:
    x: Tensor of shape [batch, ...], where ... can be any rank such that the
      number of elements in x is batch * rows * cols * hparams.hidden_size.
    rows: Integer representing number of rows in a 2-D data point.
    cols: Integer representing number of columns in a 2-D data point.
    hparams: HParams set.

  Returns:
    Tensor of shape [batch, rows, cols, depth], where depth is
    hparams.num_mixtures * 10 if hparams.likelihood is DMOL, otherwise 256. In
    the special case of inference and block raster scan order, it is a Tensor
    of shape [batch, num_blocks_rows, num_block_cols, block_length, block_width,
    depth].
  """
  batch = common_layers.shape_list(x)[0]
  x = tf.reshape(x, [batch, rows, cols, hparams.hidden_size])
  likelihood = getattr(hparams, "likelihood", cia.DistributionType.CAT)
  if likelihood == cia.DistributionType.DMOL:
    depth = hparams.num_mixtures * 10
    targets = tf.layers.dense(x,
                              depth,
                              use_bias=False,
                              activation=None,
                              name="output_conv")
  else:
    depth = target_vocab_size
    targets = tf.layers.dense(x,
                              depth,
                              use_bias=True,
                              activation=None,
                              name="output_conv")
  if (hparams.mode == tf.estimator.ModeKeys.PREDICT and
      hparams.block_raster_scan):
    y = targets
    yshape = common_layers.shape_list(y)
    block_length = hparams.query_shape[0]
    block_width = hparams.query_shape[1]

    # Break into block row wise.
    y = tf.reshape(y,
                   [batch, yshape[1] // block_length, block_length,
                    yshape[2], depth])
    yshape = common_layers.shape_list(y)
    # Break into blocks width wise.
    y_blocks = tf.reshape(y,
                          [batch, yshape[1], yshape[2],
                           yshape[3] // block_width, block_width, depth])

    # Reshape targets as [batch, num_blocks_rows, num_block_cols, block_length,
    # block_width, depth].
    targets = tf.transpose(y_blocks, [0, 1, 3, 2, 4, 5])

  return targets

def custom_create_output(decoder_output, rows, cols, targets, hparams, target_vocab_size):
  """Creates output from decoder output and vars.

  Args:
    decoder_output: Tensor of shape [batch, ...], where ... can be any rank such
      that the number of elements is batch * rows * cols * hparams.hidden_size.
    rows: Integer representing number of rows in a 2-D data point.
    cols: Integer representing number of columns in a 2-D data point.
    targets: Tensor of shape [batch, hparams.img_len, hparams.img_len,
      hparams.num_channels].
    hparams: HParams set.

  Returns:
    Tensor of shape [batch, hparams.img_len, hparams.img_len,
    hparams.num_mixtures * 10] if hparams.likelihood is DMOL, otherwise
    [batch, hparams.img_len, hparams.img_len, hparams.num_channels, 256].
    In the special case of predict mode, it is a Tensor of rank 5.
  """
  del targets  # unused arg
  decoded_image = custom_postprocess_image(decoder_output, rows, cols, hparams, target_vocab_size)
  batch = common_layers.shape_list(decoded_image)[0]
  depth = common_layers.shape_list(decoded_image)[-1]
  likelihood = getattr(hparams, "likelihood", cia.DistributionType.CAT)
  if hparams.mode == tf.estimator.ModeKeys.PREDICT:
    y = tf.reshape(decoded_image, [batch, -1, 1, 1, depth])
    output = y[:, :rows, :, :, :]
  elif likelihood == cia.DistributionType.CAT:
    # Unpack the cols dimension of the Categorical.
    channels = hparams.target_num_channels
    output = tf.reshape(decoded_image,
                        [batch, rows, cols // channels, channels, depth])
  else:
    output = decoded_image
  return output

@registry.register_model
class Imgs2imgTransformer(t2t_model.T2TModel):
  """Multiple Images 2 Image transformer net."""

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               data_parallelism=None,
               decode_hparams=None,
               **kwargs):

    super(Imgs2imgTransformer, self).__init__(
      hparams,
      mode,
      problem_hparams,
      data_parallelism,
      decode_hparams,
      **kwargs)

    self.input_conv = tf.keras.layers.Conv2D(
      filters=hparams.hidden_size,
      kernel_size=(1, 1),
      padding="same"
    )

    self.target_conv = tf.keras.layers.Conv2D(
      filters=hparams.hidden_size,
      kernel_size=(1, 1),
      padding="same"
    )

    self.output_conv = tf.keras.layers.Conv2D(
      filters=1,
      kernel_size=(1, 1),
      padding="same"
    )



  def bottom(self, features):
    transformed_features = collections.OrderedDict()
    if "inputs" in features:
      transformed_features["inputs"] = self.input_conv(features["inputs"])

    if "targets" in features:
      transformed_features["targets"] = self.target_conv(features["targets"])
    return transformed_features


  def body(self, features):
    hparams = copy.copy(self._hparams)
    targets = features["targets"]
    inputs = features["inputs"]
    # if not (tf.get_variable_scope().reuse or
    #         hparams.mode == tf.estimator.ModeKeys.PREDICT):
    #   tf.summary.image("inputs", inputs, max_outputs=1)
    #   tf.summary.image("targets", targets, max_outputs=1)
    # inputs.set_shape((None, hparams.example_shape, hparams.example_shape * hparams.num_channels, hparams.hidden_size))
    # print("Body Input", inputs.shape)
    # inputs_shape = inputs.shape
    # inputs = tf.reshape(inputs, [self._hparams.batch_size, -1, inputs_shape[2], inputs_shape[3]])
    # print("Reshaped Inputs", inputs.shape)
    encoder_input = cia.prepare_encoder(inputs, hparams, hparams.enc_attention_type)
    # print("Encoder Input", encoder_input.shape)
    encoder_output = cia.transformer_encoder_layers(
        encoder_input,
        hparams.num_encoder_layers,
        hparams,
        attention_type=hparams.enc_attention_type,
        name="encoder")
    # print("Encoder Output", encoder_output.shape)
    # print("Body Targets", targets.shape)
    decoder_input, rows, cols = cia.prepare_decoder(
        targets, hparams)
    # print("Decoder Input", decoder_input.shape)
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_decoder_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        name="decoder")
    # print("Decoder Output", decoder_output.shape)
    output = self.output_conv(decoder_output)
    # output = custom_create_output(decoder_output, rows, cols, targets, hparams, self._problem_hparams.vocab_size["targets"])
    # output = tf.squeeze(output, axis=-1)
    print("Final Output", output)
    return output


def custom_image_transformer2d_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 1
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = .0  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 0.2
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  # hparams.bottom["targets"] = modalities.identity_bottom
  hparams.bottom["targets"] = modalities.symbol_bottom
  # hparams.bottom["targets"] = modalities.make_targets_bottom(
  #     modalities.image_channel_embeddings_bottom)
  hparams.top["targets"] = modalities.identity_top
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.add_hparam("filter_size", 512)  # Add new ones like this.

  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("num_output_layers", 3)
  hparams.add_hparam("block_size", 1)

  # image size related flags
  # assuming that the image has same height and width
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  hparams.add_hparam("target_num_channels", 1)
  # Local attention params
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 256)
  hparams.add_hparam("block_width", 128)
  # Local 2D attention params
  hparams.add_hparam("query_shape", (8, 8))
  hparams.add_hparam("memory_flange", (8, 8))
  hparams.add_hparam("num_encoder_layers", 4)
  hparams.add_hparam("num_decoder_layers", 8)
  # attention type related params
  # hparams.add_hparam("enc_attention_type", cia.AttentionType.GLOBAL)
  hparams.add_hparam("enc_attention_type", cia.AttentionType.LOCAL_2D)
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_2D)
  hparams.add_hparam("block_raster_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("unconditional", False)  # unconditional generation

  # relative embedding hparams
  hparams.add_hparam("shared_rel", False)
  return hparams


@registry.register_hparams
def custom_img2img_transformer2d_base():
  """Base params for img2img 2d attention."""
  hparams = custom_image_transformer2d_base()
  # learning related flags
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  # This version seems to benefit from a higher learning rate.
  hparams.learning_rate = 0.2
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 12000
  hparams.filter_size = 2048
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 8
  # hparams.bottom["inputs"] = modalities.identity_bottom
  hparams.bottom["inputs"] = modalities.image_channel_embeddings_bottom
  hparams.dec_attention_type = cia.AttentionType.LOCAL_2D
  hparams.block_raster_scan = True
  return hparams

@registry.register_hparams
def custom_img2img_transformer2d_tiny():
  """Tiny params."""
  hparams = custom_img2img_transformer2d_base()
  hparams.num_decoder_layers = 2
  hparams.hidden_size = 128
  hparams.batch_size = 5
  hparams.max_length = 128
  hparams.attention_key_channels = hparams.attention_value_channels = 0
  hparams.filter_size = 128
  hparams.num_heads = 4
  hparams.pos = "timing"
  hparams.img_len = 32
  hparams.query_shape = (8, 8)
  hparams.memory_flange = (8, 8)
  return hparams

class SuperResoProblem(image_utils.ImageProblem):
  """Test problem."""

  def __init__(self, input_vocab_size, target_vocab_size):
     super(image_utils.ImageProblem, self).__init__()
     #super(TestProblem, self).__init__(False, False)
     self.input_vocab_size = input_vocab_size
     self.target_vocab_size = target_vocab_size

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    # p.modality = {"inputs": modalities.ModalityType.IMAGE,
    #               "targets": modalities.ModalityType.IMAGE}
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": self.input_vocab_size,
                    "targets": self.target_vocab_size}
    p.input_space_id = 1
    p.target_space_id = 1


def super_reso_problem_hparams(input_vocab_size, target_vocab_size, model_hparams=None):
  """Problem hparams for testing model bodies."""
  p = SuperResoProblem(input_vocab_size, target_vocab_size)
  return p.get_hparams(model_hparams)
