"""convolution neural net training"""

import os, sys
import argparse
import numpy as np
import tensorflow as tf

parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)

import picar

FLAGS = None
NUM_CHANNELS = 3
NUM_LABELS = 3
BATCH_SIZE = 20
NUM_EPOCH = 8


def RestoreChannels(images):
  """ Reshape images, whose shape is (num samples, 3 * width * height) to
      shape (num samples, width, height, channels).
  """
  out = np.zeros(shape=(len(images), picar.RESIZE, picar.RESIZE, NUM_CHANNELS))
  # Should have a better way to do this ...
  for i in xrange(len(images)):
    im = images[i]
    for x in range(picar.RESIZE):
      for y in range(picar.RESIZE):
        idx = NUM_CHANNELS * (x * picar.RESIZE + y)
        for c in range(NUM_CHANNELS):
          out[i, x, y, c] = im[idx + c]
  return out


class CnnWeights(object):
  def __init__(self):
    self.conv_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            dtype=tf.float32))
    self.conv_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))

    self.fc_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([picar.RESIZE // 2 * picar.RESIZE // 2 * 32, NUM_LABELS],
                            stddev=0.1,
                            dtype=tf.float32))
    self.fc_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))


class CnnModel(object):
  """convolution network model."""

  def __init__(self, weights, images_data, labels):
    
    # Construct the graph
    # (1) convolution layer
    conv = tf.nn.conv2d(images_data,
                        weights.conv_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, weights.conv_biases))

    # (2) Max pooling.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

    # Fully connected layer.
    logits = tf.matmul(reshape, weights.fc_weights) + weights.fc_biases
    self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, labels))
    self._train_step = tf.train.MomentumOptimizer(0.008, 0.9).minimize(self._loss)

    # For validation set
    predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    self._accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

  @property
  def train_step(self):
    return self._train_step

  @property
  def loss(self):
    return self._loss

  @property
  def accuracy(self):
    return self._accuracy


def main(_):
  picar_data = picar.read_data_sets(FLAGS.data_dir)

  # Make each images of shape (channel, width, height)
  picar_data.train._images = RestoreChannels(picar_data.train._images)
  picar_data.validation._images = RestoreChannels(picar_data.validation._images)

  # Place holders for training images and labels
  train_data = tf.placeholder(
      tf.float32,
      shape=(BATCH_SIZE, picar.RESIZE, picar.RESIZE, NUM_CHANNELS))
  train_labels = tf.placeholder(tf.int64, shape=(BATCH_SIZE, NUM_LABELS))

  # Place holders for validation images and labels
  eval_data = tf.placeholder(
      tf.float32,
      shape=(picar_data.validation.num_examples, picar.RESIZE, picar.RESIZE, NUM_CHANNELS))
  eval_labels = tf.placeholder(tf.int64, shape=(picar_data.validation.num_examples, NUM_LABELS))

  weights = CnnWeights()

  # Cnn graph for training uses
  training = CnnModel(weights, train_data, train_labels)

  # Cnn graph, with the same weights as above, for prediction uses
  eval = CnnModel(weights, eval_data, eval_labels)

  sess = tf.InteractiveSession()
  tf.initialize_all_variables().run()

  for epoch in range(NUM_EPOCH):
    for b in range(picar_data.train.num_examples / BATCH_SIZE):
      images, labels = picar_data.train.next_batch(BATCH_SIZE)
      val = sess.run([training.train_step, training.loss],
                     feed_dict = {train_data: images, train_labels: labels})
    print "training loss at end of epoch ", epoch, " : ", val[1]

  accuracy = sess.run(eval.accuracy,
                      feed_dict = {eval_data: picar_data.validation.images,
                                   eval_labels: picar_data.validation.one_hot_labels})
  print "prediction accuracy on validation set: ", accuracy


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='../../../data/round1',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
