from analysis import plot_utils, recorder_utils
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def summarize_template_1(
  args,
  observation_spec,
  hparams,
  steps,
  tb_dir,
  save_dir
):
  tensor_tags = ["predictions/distribution_tensor", "predictions/predictions_tensor",
  "predictions/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]

  templater = recorder_utils.SlideTemplater(True, args.service_account_path, args.slide_id)

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
    text_1 = [os.path.basename(args.job_dir)]
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss', 'accuracy'],
        step)] + [hparams.bets, hparams.rewards, str(hparams.scale_steps)]

    angles = ", ".join(['{:.3g}'.format(elem) for elem in observation_spec.angles])
    frequencies = ", ".join(['{:.3g}'.format(elem.frequency) for elem in observation_spec.psf_descriptions])
    numerical_aperture = '{:.3g}'.format(observation_spec.psf_descriptions[0].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(observation_spec.psf_descriptions[0].frequency_sigma)
    num_angles = len(args.angle_indices)
    num_frequencies = len(args.frequency_indices)
    num_total = num_angles * num_frequencies
    text_3 = [numerical_aperture, str(num_total), str(num_angles), str(num_frequencies)]

    example_dist = plot_utils.get_tensors_from_tensorboard(tb_dir, ["predictions/distribution_tensor"],
      step)[0]
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
    label_4 = ['Bit depth', 'Obs height', 'Obs width', 'Pred height', 'Pred width']
    label_5 = ['Encoders', 'Decoders', 'Enc-Dec heads', 'Hidden size', 'Dropout']
    labels = [label_1, label_2, label_3, label_4, label_5]

    comment = hparams.to_json() + "Angles: " + angles + "Frequencies: " + frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()

def summarize_template_2(
  args,
  observation_spec,
  hparams,
  steps,
  tb_dir,
  save_dir
):
  tensor_tags = ["predictions/distribution_tensor", "predictions/predictions_tensor",
  "predictions/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]

  templater = recorder_utils.SlideTemplater(True, args.service_account_path, args.slide_id)

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
    text_1 = [os.path.basename(args.job_dir)]
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss', 'accuracy'],
        step)] + [hparams.bets, hparams.rewards, str(hparams.diff_scale)]

    angles = ", ".join(['{:.3g}'.format(elem) for elem in observation_spec.angles])
    frequencies = ", ".join(['{:.3g}'.format(elem.frequency) for elem in observation_spec.psf_descriptions])
    numerical_aperture = '{:.3g}'.format(observation_spec.psf_descriptions[0].numerical_aperture)
    frequency_sigma = '{:.3g}'.format(observation_spec.psf_descriptions[0].frequency_sigma)
    num_angles = len(args.angle_indices)
    num_frequencies = len(args.frequency_indices)
    num_total = num_angles * num_frequencies
    text_3 = [numerical_aperture, str(num_total), str(num_angles), str(num_frequencies)]

    example_dist = plot_utils.get_tensors_from_tensorboard(tb_dir, ["predictions/distribution_tensor"],
      step)[0]
    obs_height = args.example_shape[0] / hparams.observation_pool_downsample
    obs_width = args.example_shape[1] / hparams.observation_pool_downsample
    text_4 = [str(hparams.bit_depth), str(int(obs_height)), str(int(obs_width)),
      str(example_dist.shape[1]), str(example_dist.shape[2])]
    text_5 = [str(hparams.unet_height), str(hparams.initial_filters), hparams.embedding, hparams.unet_type, str(hparams.recurrent)
      , str(hparams.num_split)]
    text_6 = ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['accuracy_weighted', 'non_zero_acc',
      'precision', 'recall', 'f1'], step)]

    texts = [text_1, text_2, text_3, text_4, text_5, text_6]

    label_1 = None
    label_2 = ['Steps', 'Loss', 'Accuracy', "Bets", "Rewards", "Diff scale"]
    label_3 = ['NA', 'Observed', 'Angles', 'Freqs']
    label_4 = ['Bit depth', 'Obs height', 'Obs width', 'Pred height', 'Pred width']
    label_5 = ['UNet height', 'Initial filters', 'Embedding', 'Unet type', 'Recurrent','Num split']
    label_6 = ['Weighted acc', 'Non-zero acc', 'Precision', 'Recall', 'F1']
    labels = [label_1, label_2, label_3, label_4, label_5, label_6]

    comment = hparams.to_json() + "Angles: " + angles + "Frequencies: " + frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()

def summarize_template_3(
  args,
  dataset_params,
  simulation_params,
  model_params,
  steps,
  tb_dir,
  save_dir
):
  tensor_tags = ["predictions/distribution_tensor", "predictions/predictions_tensor",
  "predictions/difference_tensor"]
  titles = ["distribution", "prediction", "difference"]

  templater = recorder_utils.SlideTemplater(True, args.service_account_path, args.slide_id)

  steps.reverse()
  save_paths = [[] for _ in range(len(steps))]
  for s, step in enumerate(steps):
    figs = plot_utils.plot_grid_from_tensorboard(tb_dir,
      tensor_tags, step, titles=titles, scale=dataset_params.grid_dimension * 16)

    for i in range(len(figs)):
      save_paths[s].append(save_dir + "/eval/pred_{}_{}.png".format(step, i+1))

    for fig, b_path in zip(figs, save_paths[s]):
        templater.storage.upload_maybe_fig(fig, b_path)
        plt.close(fig)

  for save_path, step in zip(save_paths, steps):
    model_params.del_hparam("problem")
    text_1 = [os.path.basename(args.job_dir)]
    text_2 = [str(step)] + ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['loss', 'accuracy'],
        step)] + [model_params.bets, model_params.rewards, str(model_params.diff_scale)]

    angles_float = np.linspace(0., args.angle_limit, args.angle_count)
    frequencies_float = np.linspace(args.min_frequency, args.max_frequency, args.frequency_count)
    angles = ", ".join(['{:.3g}'.format(elem) for elem in angles_float])
    frequencies = ", ".join(['{:.3g}'.format(elem) for elem in frequencies_float])
    numerical_aperture = '{:.3g}'.format(simulation_params.numerical_aperture)
    frequency_sigma = '{:.3g}'.format(simulation_params.frequency_sigma)
    num_angles = args.angle_count
    num_frequencies = args.frequency_count
    num_total = num_angles * num_frequencies
    text_3 = [numerical_aperture, str(num_total), str(num_angles), str(num_frequencies)]

    example_dist = plot_utils.get_tensors_from_tensorboard(tb_dir, ["predictions/distribution_tensor"],
      step)[0]
    text_4 = [str(model_params.bit_depth)]
    text_5 = [str(model_params.unet_height), str(model_params.initial_filters),
      model_params.embedding, model_params.unet_type, str(model_params.recurrent)
      , str(model_params.num_split)]
    text_6 = ['{:.3g}'.format(elem) for elem in
      plot_utils.get_scalars_from_tensorboard(tb_dir,['accuracy_weighted', 'non_zero_acc',
      'precision', 'recall', 'f1'], step)]

    texts = [text_1, text_2, text_3, text_4, text_5, text_6]

    label_1 = None
    label_2 = ['Steps', 'Loss', 'Accuracy', "Bets", "Rewards", "Diff scale"]
    label_3 = ['NA', 'Observed', 'Angles', 'Freqs']
    label_4 = ['Bit depth']
    label_5 = ['UNet height', 'Initial filters', 'Embedding', 'Unet type', 'Recurrent','Num split']
    label_6 = ['Weighted acc', 'Non-zero acc', 'Precision', 'Recall', 'F1']
    labels = [label_1, label_2, label_3, label_4, label_5, label_6]

    comment = model_params.to_json() + dataset_params.to_json() + \
      simulation_params.to_json() + "Angles: " + angles + "Frequencies: " + \
      frequencies + "Frequency Sigma: " + frequency_sigma

    templater.fill_template_from_cloud(texts, save_path, labels, comment=comment)

  templater.execute()
