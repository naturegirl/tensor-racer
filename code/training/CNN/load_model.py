"""load saved CNN model and do predictions"""

import argparse
import os, sys
import numpy as np
import tensorflow as tf

parent_dir = os.path.abspath("..")
sys.path.insert(0, parent_dir)
import picar

FLAGS = None

def main(_):
    picar_data = picar.read_data_sets(FLAGS.data_dir, flatten=False)
    meta_file = FLAGS.model_file + ".meta"
    saver = tf.train.import_meta_graph(meta_file)
    sess = tf.InteractiveSession()
    saver.restore(sess, FLAGS.model_file)

    print("done restoring model")
    predictions = tf.get_collection("predictions")
    data = tf.get_collection("prediction_data")[0]
    for img in picar_data.validation.images:
        pred = sess.run(predictions,
                        feed_dict={data: np.array([img])})[0][0]
        print "predicted label:", pred

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='../../../data/round1',
                      help='Directory for storing data')
  parser.add_argument('--model_file', type=str, default='',
                      help='file path to save a trained model')
  FLAGS = parser.parse_args()
  tf.app.run()
