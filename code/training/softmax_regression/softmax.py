"""softmax regression implementation with SGD"""

# Code is heavily based on:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/picar/picar_softmax.py


import argparse
import tensorflow as tf
import os, sys

# add parent dir to be able to retrieve picar.py
parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)

from picar import read_data_sets

FLAGS = None
NUM_PIXELS = 2700
NUM_LABELS = 3

def main(_):
  picar = read_data_sets(FLAGS.data_dir)

  # Create the model
  x = tf.placeholder(tf.float32, [None, NUM_PIXELS])
  W = tf.Variable(tf.zeros([NUM_PIXELS, NUM_LABELS]))
  b = tf.Variable(tf.zeros([NUM_LABELS]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.sparse_softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(5 * 400 / 20):
    batch_xs, batch_ys = picar.train.next_batch(20)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: picar.validation.images,
                                      y_: picar.validation.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='../../../data/round1',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
