from analysis import plot_utils, recorder_utils
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from online_simulation import online_simulation_utils

def summarize_template_1(
  args,
  observation_spec,
  hparams,
  steps,
  tb_dir,
  save_dir
):
  '''
  Template for Image Transformer model. Creates Google Slides that summarize
  evaluation results.

  Args:
    args: User's arguments passed to train script.
    observation_spec: Observation spec used in simulation.
    hparams: Model's hyperparameters.
    steps: List of ints denoting steps to summarize.
    tb_dir: Tensorboard directory.
    save_dir: Google Storage directory to save plotted figures.
  '''

  tensor_tags = ["predictions/distribution_tensor",
    "predictions/predictions_tensor", "predictions/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]

  templater = recorder_utils.SlideTemplater(True, args.service_account_path,
    args.slide_id)

  # reverse steps to ensure proper ordering as slides created first appear
  # last to facilitate proper batching.
  steps.reverse()
  save_paths = [[] for _ in range(len(steps))]
  for s, step in enumerate(steps):
    figs = plot_utils.plot_grid_from_tensorboard(tb_dir,
      tensor_tags, step, titles=titles, scale=observation_spec.grid_dimension *
      hparams.observation_pool_downsample)

    for i in range(len(figs)):
      save_paths[s].append(save_dir + "/eval/pred_{}_{}.png".format(step, i+1))

    for fig, b_path in zip(figs, save_paths[s]):
        templater.storage.upload_maybe_fig(fig, b_path)
        plt.close(fig)

  for save_path, step in zip(save_paths, steps):
    text_1 = [os.path.basename(args.job_dir)]
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss', 'accuracy'],
        step)] + [hparams.bets, hparams.rewards, str(hparams.scale_steps)]

    angles = ", ".join(['{:.3g}'.format(elem) for elem in
      observation_spec.angles])
    frequencies = ", ".join(['{:.3g}'.format(elem.frequency) for elem in
      observation_spec.psf_descriptions])

    # assuming same NA and frequency_sigma for all observations.
    numerical_aperture = '{:.3g}'.format(
      observation_spec.psf_descriptions[0].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(
      observation_spec.psf_descriptions[0].frequency_sigma)

    num_angles = len(args.angle_indices)
    num_frequencies = len(args.frequency_indices)
    num_total = num_angles * num_frequencies
    text_3 = [numerical_aperture, str(num_total), str(num_angles),
      str(num_frequencies)]

    example_dist = plot_utils.get_tensors_from_tensorboard(tb_dir,
      ["predictions/distribution_tensor"], step)[0]
    obs_height = args.example_shape[0] / hparams.observation_pool_downsample
    obs_width = args.example_shape[1] / hparams.observation_pool_downsample
    text_4 = [str(hparams.bit_depth), str(int(obs_height)), str(int(obs_width)),
      str(example_dist.shape[1]), str(example_dist.shape[2])]
    text_5 = [str(hparams.num_encoder_layers), str(hparams.num_decoder_layers),
      str(hparams.num_heads), str(hparams.hidden_size), str(hparams.dropout)]
    texts = [text_1, text_2, text_3, text_4, text_5]

    label_1 = None
    label_2 = ['Steps', 'Loss', 'Accuracy', "Bets", "Rewards", "Scale steps"]
    label_3 = ['NA', 'Observed', 'Angles', 'Freqs']
    label_4 = ['Bit depth', 'Obs height', 'Obs width', 'Pred height',
      'Pred width']
    label_5 = ['Encoders', 'Decoders', 'Enc-Dec heads', 'Hidden size',
      'Dropout']
    labels = [label_1, label_2, label_3, label_4, label_5]

    comment = hparams.to_json() + "Angles: " + angles + "Frequencies: " + \
      frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels,
      comment=comment)

  templater.execute()


def summarize_model_builder(
  args,
  observation_spec,
  hparams,
  steps,
  tb_dir,
  save_dir
):
  '''
  Template for Model Builder. Creates Google Slides that summarize evaluation
  results.

  Args:
    args: User's arguments passed to train script.
    observation_spec: Observation spec used in simulation.
    hparams: Model's hyperparameters.
    steps: List of ints denoting steps to summarize.
    tb_dir: Tensorboard directory.
    save_dir: Google Storage directory to save plotted figures.
  '''

  tensor_tags = ["predictions/distribution_tensor",
    "predictions/predictions_tensor", "predictions/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]

  templater = recorder_utils.SlideTemplater(True, args.service_account_path,
    args.slide_id)

  # reverse steps to ensure proper ordering as slides created first appear
  # last to facilitate proper batching.
  steps.reverse()
  save_paths = [[] for _ in range(len(steps))]
  for s, step in enumerate(steps):
    figs = plot_utils.plot_grid_from_tensorboard(tb_dir,
      tensor_tags, step, titles=titles, scale=observation_spec.grid_dimension)

    for i in range(len(figs)):
      save_paths[s].append(save_dir + "/eval/pred_{}_{}.png".format(step, i+1))

    for fig, b_path in zip(figs, save_paths[s]):
        templater.storage.upload_maybe_fig(fig, b_path)
        plt.close(fig)

  for save_path, step in zip(save_paths, steps):
    hparams.del_hparam("problem")

    # Job
    text_1 = [os.path.basename(args.job_dir)]
    label_1 = None

    # Loss / Training
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss'],
        step)] + [hparams.bets, hparams.rewards, str(hparams.diff_scale)]
    label_2 = ['Steps', 'Loss', "Bets", "Rewards", "Diff scale"]
    if "recurrent" in hparams.recurrent:
      text_2 += [str(hparams.last_loss_only), str(hparams.loss_scale)]
      label_2 += ['Last loss only', 'Loss scale']

    # Metrics
    text_3 = ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['accuracy',
      'accuracy_weighted', 'non_zero_acc', 'precision', 'recall', 'f1'], step)]
    label_3 = ['Acc', 'Weighted acc', 'Non-zero acc', 'Precision', 'Recall',
      'F1']

    # Simulation
    angles = ", ".join(['{:.3g}'.format(elem) for elem in
      observation_spec.angles])
    frequencies = ", ".join(['{:.3g}'.format(elem.frequency) for
      elem in observation_spec.psf_descriptions])
    numerical_aperture = '{:.3g}'.format(
      observation_spec.psf_descriptions[0].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(
      observation_spec.psf_descriptions[0].frequency_sigma)
    num_angles = len(args.angle_indices)
    num_frequencies = len(args.frequency_indices)
    num_total = num_angles * num_frequencies

    text_4 = [numerical_aperture, str(num_total), str(num_angles),
      str(num_frequencies), str(hparams.bit_depth)]
    label_4 = ['NA', 'Observed', 'Angles', 'Freqs', 'Bit depth']

    # Model
    text_5 = [hparams.model_type, hparams.recurrent, hparams.embedding,
      hparams.pooler, str(hparams.dropout)]
    label_5 = ['Type', 'Recur', 'Emb', 'Pooler', 'Dropout']
    if 'recurrent' in hparams.recurrent:
      text_5 += [str(hparams.channels_per_step), hparams.gating]
      label_5 += ['Channels/step', 'Gating']

    # Model extended
    text_6 = []
    label_6 = []
    if hparams.model_type == 'unet':
      text_6 += [str(hparams.forward_height) +'/'+ str(hparams.reverse_height),
        hparams.forward_prop, str(hparams.forward_kwargs),
        hparams.mid_prop, str(hparams.mid_kwargs), hparams.reverse_prop,
        str(hparams.reverse_kwargs), hparams.forward_conv_name,
        hparams.reverse_conv_name, str(hparams.unet_attention)]
      label_6 += ['Unet heights', 'Forward prop', 'Forward args', 'Mid prop',
        'Mid args', 'Reverse prop', 'Reverse args', 'Forward conv',
        'Reverse conv', 'Attention']
    else:
      text_6 += [str(hparams.prop_layers), hparams.forward_prop,
        str(hparams.forward_kwargs), hparams.forward_conv_name]
      label_6 += ['Prop layers', 'Propagator', 'Args', 'Pooling Conv']

    texts = [text_1, text_2, text_3, text_4, text_5, text_6]
    labels = [label_1, label_2, label_3, label_4, label_5, label_6]

    comment = hparams.to_json() + "Angles: " + angles + \
      "Frequencies: " + frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()


def summarize_online_builder(
  args,
  dataset_params,
  simulation_params,
  model_params,
  steps,
  tb_dir,
  save_dir
):
  '''
  Template for Online Model Builder. Creates Google Slides that summarize
  evaluation results.

  Args:
    args: User's arguments passed to train script.
    dataset_params: HParams governing dataset generation.
    simulation_params: HParams governing simulation.
    model_params: HParams of model.
    steps: List of ints denoting steps to summarize.
    tb_dir: Tensorboard directory.
    save_dir: Google Storage directory to save plotted figures.
  '''

  tensor_tags = ["features/images/distribution_tensor",
    "features/images/predictions_tensor", "features/images/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]
  DOWNSAMPLE = 2 ** model_params.downsample_bits

  templater = recorder_utils.SlideTemplater(True, args.service_account_path,
    args.slide_id)

  steps.reverse()
  save_paths = [[] for _ in range(len(steps))]

  for s, step in enumerate(steps):
    figs = plot_utils.plot_grid_from_tensorboard(tb_dir, tensor_tags, step,
      titles=titles, scale=dataset_params.grid_dimension * DOWNSAMPLE, limit=4)

    for i in range(len(figs)):
      save_paths[s].append(save_dir + "/eval/pred_{}_{}.png".format(step, i+1))

    for fig, b_path in zip(figs, save_paths[s]):
        templater.storage.upload_maybe_fig(fig, b_path)
        plt.close(fig)

  for save_path, step in zip(save_paths, steps):

    # Job
    text_1 = [os.path.basename(args.job_dir)]
    label_1 = None

    # Loss / Training
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss'],
        step)] + [model_params.bets, model_params.rewards,
        str(model_params.diff_scale)]
    label_2 = ['Steps', 'Loss', "Bets", "Rewards", "Diff scale"]
    if "recurrent" in model_params.recurrent:
      text_2 += [str(model_params.last_loss_only), str(model_params.loss_scale)]
      label_2 += ['Last loss only', 'Loss scale']

    # Metrics
    text_3 = ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['accuracy',
      'accuracy_weighted', 'non_zero_acc', 'mean_squared_error', 'precision',
      'recall', 'f1'], step)]
    label_3 = ['Acc', 'Weighted acc', 'Non-zero acc', 'MSE', 'Precision',
      'Recall', 'F1']

    # Simulation
    angles = ", ".join(['{:.3g}'.format(elem[0]) for elem in
      simulation_params.psf_descriptions])
    frequencies = ", ".join(['{:.3g}'.format(elem[1].frequency) for
      elem in simulation_params.psf_descriptions])
    modes = ", ".join(['{:.3g}'.format(elem[1].mode) for
      elem in simulation_params.psf_descriptions])
    numerical_aperture = '{:.3g}'.format(
      simulation_params.psf_descriptions[0][1].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(
      simulation_params.psf_descriptions[0][1].frequency_sigma)
    num_angles = args.angle_count
    num_frequencies = args.frequency_count
    num_modes = args.mode_count
    num_total = num_angles * num_frequencies * num_modes

    text_4 = [numerical_aperture, str(num_total), str(num_angles),
      str(num_frequencies), str(num_modes), str(args.angle_limit),
      '{:.3g}'.format(dataset_params.scatterer_density),
      '{:.3g}'.format(model_params.lambda_multiplier),
      str(model_params.bit_depth)]
    label_4 = ['NA', 'Observed', 'Angles', 'Freqs', 'Modes', 'Angle limit',
      'Density', 'Bit depth']

    # Model
    text_5 = [model_params.model_type, model_params.recurrent, model_params.embedding,
      model_params.pooler, str(model_params.dropout), str(model_params.concat_avg)]
    label_5 = ['Type', 'Recur', 'Emb', 'Pooler', 'Dropout', 'Concat avg']
    if 'recurrent' in model_params.recurrent:
      text_5 += [str(model_params.channels_per_step), model_params.gating]
      label_5 += ['Channels/step', 'Gating']

    # Model extended
    text_6 = []
    label_6 = []
    if model_params.model_type == 'unet':
      text_6 += [str(model_params.forward_height) +'/'+ str(model_params.reverse_height),
        model_params.forward_prop, str(model_params.forward_kwargs),
        model_params.mid_prop, str(model_params.mid_kwargs), model_params.reverse_prop,
        str(model_params.reverse_kwargs), model_params.forward_conv_name,
        model_params.reverse_conv_name, str(model_params.unet_attention)]
      label_6 += ['Unet heights', 'Forward prop', 'Forward args', 'Mid prop',
        'Mid args', 'Reverse prop', 'Reverse args', 'Forward conv',
        'Reverse conv', 'Attention']
    else:
      text_6 += [str(model_params.prop_layers), str(model_params.pooler_filters),
        model_params.forward_prop, str(model_params.forward_kwargs),
        model_params.forward_conv_name]
      label_6 += ['Prop layers', 'Filters', 'Propagator', 'Args', 'Pooling Conv']

    texts = [text_1, text_2, text_3, text_4, text_5, text_6]
    labels = [label_1, label_2, label_3, label_4, label_5, label_6]

    comment = model_params.to_json() + dataset_params.to_json() + \
      simulation_params.to_json() + "Angles: " + angles + \
      "Frequencies: " + frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()


def summarize_online_simulation(
  args,
  dataset_params,
  simulation_params,
  model_params,
  steps,
  tb_dir,
  save_dir
):
  '''
  Template for online_simulation model. Creates Google Slides that summarize
  evaluation results.

  Args:
    args: User's arguments passed to train script.
    dataset_params: HParams governing dataset generation.
    simulation_params: HParams governing simulation.
    model_params: HParams of model.
    steps: List of ints denoting steps to summarize.
    tb_dir: Tensorboard directory.
    save_dir: Google Storage directory to save plotted figures.
  '''

  DOWNSAMPLE = 2 ** model_params.downsample_bits
  tensor_tags = ["features/images/distribution_tensor",
    "features/images/predictions_tensor", "features/images/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]
  if model_params.objective == "PROBABILITY":
    titles[0] = "probability"

  templater = recorder_utils.SlideTemplater(True, args.service_account_path, args.slide_id)

  steps.reverse()
  save_paths = [[] for _ in range(len(steps))]

  for s, step in enumerate(steps):
    figs = plot_utils.plot_grid_from_tensorboard(tb_dir,
      tensor_tags, step, titles=titles,
      scale=dataset_params.grid_dimension * DOWNSAMPLE, limit=4)

    for i in range(len(figs)):
      save_paths[s].append(save_dir + "/eval/pred_{}_{}.png".format(step, i+1))

    for fig, b_path in zip(figs, save_paths[s]):
        templater.storage.upload_maybe_fig(fig, b_path)
        plt.close(fig)

  for save_path, step in zip(save_paths, steps):

    # Job
    text_1 = [os.path.basename(args.job_dir)]
    label_1 = None

    # Loss / Training
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss'],
        step)]
    label_2 = ['Steps', 'Loss']

    # Metrics
    text_3 = ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['accuracy_no_weight',
      'accuracy_weight', 'mean_squared_error'], step)]
    label_3 = ['Acc', 'Weighted acc', 'MSE']

    # Simulation
    angles = ", ".join(['{:.3g}'.format(elem[0]) for elem in
      simulation_params.psf_descriptions])
    frequencies = ", ".join(['{:.3g}'.format(elem[1].frequency) for
      elem in simulation_params.psf_descriptions])
    modes = ", ".join(['{:.3g}'.format(elem[1].mode) for
      elem in simulation_params.psf_descriptions])
    numerical_aperture = '{:.3g}'.format(
      simulation_params.psf_descriptions[0][1].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(
      simulation_params.psf_descriptions[0][1].frequency_sigma)
    num_angles = args.angle_count
    num_frequencies = args.frequency_count
    num_modes = args.mode_count
    num_total = num_angles * num_frequencies * num_modes

    text_4 = [numerical_aperture, str(num_total), str(num_angles),
      str(num_frequencies), str(num_modes), str(args.angle_limit),
      '{:.3g}'.format(dataset_params.scatterer_density),
      '{:.3g}'.format(model_params.lambda_multiplier),
      str(model_params.bit_depth)]
    label_4 = ['NA', 'Observed', 'Angles', 'Freqs', 'Modes', 'Angle limit',
      'Density', 'Bit depth']

    # Model
    text_5 = [str(model_params.squeeze_excite)]
    label_5 = ['Squeeze-excite']

    texts = [text_1, text_2, text_3, text_4, text_5]
    labels = [label_1, label_2, label_3, label_4, label_5]

    comment = model_params.to_json() + dataset_params.to_json() + \
      simulation_params.to_json() + "Angles: " + angles + \
      "Frequencies: " + frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()
