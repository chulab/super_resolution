"""Common metrics to compare images."""

import tensorflow as tf


def ssim(
    image_1,
    image_2,
    max_val,
    metrics_collections=None,
    updates_collections=None,
):
  with tf.name_scope("ssim"):
    ssim_loss = tf.image.ssim(image_1, image_2, max_val)

    m_r_sq, update_rsq_op = tf.metrics.mean(ssim_loss)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, m_r_sq)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_rsq_op)

    return m_r_sq, update_rsq_op


def psnr(
    image_1,
    image_2,
    max_val,
    metrics_collections=None,
    updates_collections=None,
):
  with tf.name_scope("psrn"):
    psnr_loss = tf.image.psnr(image_1, image_2, max_val)

    m_r_sq, update_rsq_op = tf.metrics.mean(psnr_loss)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, m_r_sq)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_rsq_op)

    return m_r_sq, update_rsq_op


def total_variation(
    image,
    metrics_collections=None,
    updates_collections=None,
):
  with tf.name_scope("total_variation"):
    total_variation_noise = tf.image.total_variation(image)

    m_r_sq, update_rsq_op = tf.metrics.mean(total_variation_noise)

    if metrics_collections:
        tf.add_to_collections(metrics_collections, m_r_sq)

    if updates_collections:
        tf.add_to_collections(updates_collections, update_rsq_op)

    return m_r_sq, update_rsq_op