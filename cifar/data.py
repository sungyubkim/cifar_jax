import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Sequence

# load dataset as a generator
# DO NOT FIX seed for random operation in function (e.g. flip, shuffle, etc.)
# This makes deterministic mini batch along epochs...
def load_dataset(
    ds,
    num_classes : int,
    aug: bool,
    batch_dims: Sequence[int],
  ):
  total_batch_size = np.prod(batch_dims)
  img_mean = tf.constant([[[0.5, 0.5, 0.5]]])
  img_std = tf.constant([[[0.25, 0.25, 0.25]]])

  def image_aug(batch):
    img, label = batch['image'], batch['label']
    img = tf.image.random_flip_left_right(img)
    padding = [[4,4],[4,4],[0,0]]
    img = tf.pad(img, padding, mode='REFLECT')
    img = tf.image.random_crop(img, (32, 32, 3))
    return {'image':img, 'label':label}

  def image_reshape(batch):
    img, label = batch['image'], batch['label']
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - img_mean) / img_std
    return {'image':img, 'label':label}

  def batch_reshape(batch):
    img, label = batch['image'], batch['label']
    img = tf.reshape(img, batch_dims+img.shape[1:])
    label = tf.one_hot(label, num_classes, 1., 0.)
    label = tf.reshape(label, batch_dims+label.shape[1:])
    return {'x':img, 'y':label}

  if aug:
    ds = ds.shuffle(50000)
    ds = ds.map(image_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
  ds = ds.map(image_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(total_batch_size, drop_remainder=True)
  ds = ds.map(batch_reshape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  yield from tfds.as_numpy(ds)