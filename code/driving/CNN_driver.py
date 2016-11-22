import argparse
from driver import Driver
import os, sys
import numpy as np
from sklearn.externals import joblib
import tensorflow as tf
parent_dir = os.path.abspath("..")
sys.path.insert(0, os.path.join(parent_dir, "training"))
import picar

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str,
                    default='../training/CNN/model/cnn_model',
                    help='Path fo saved tensorflow model. Should also include a meta file')
parser.add_argument('--data_dir', type=str,
                    default='../../data/round1',
                    help='Directory for storing data')
parser.add_argument('--imagefile', type=str,
                    help='Image file on which to run inference on.')
args = parser.parse_args()

class CNN_Driver(Driver):
    def __init__(self, modelfile, imagefile=None, nodrive=False, datadir=None):
        """see superclass for what the parameters mean"""
        self.datadir = datadir
        super(CNN_Driver, self).__init__(modelfile, imagefile, nodrive,
                                         flatten=False)

    def load_model(self, modelfile):
        if not os.path.exists(modelfile):
            raise Exception("model file does not exist")
        meta_file = modelfile + ".meta"
        saver = tf.train.import_meta_graph(meta_file)
        sess = tf.InteractiveSession()
        saver.restore(sess, modelfile)
        print("done restoring model")
        self.predictions = tf.get_collection("predictions")
        self.data = tf.get_collection("prediction_data")[0]
        data = tf.get_collection("prediction_data")[0]
        return sess

    def predict(self, x):
        """make predictions given model and feature vector x"""
        img = np.array([x])
        y = self.model.run(self.predictions,
                           feed_dict={self.data: img})[0][0]
        print "y", y
        return y

d = CNN_Driver(modelfile=args.modelfile, imagefile=args.imagefile, nodrive=False,
               datadir=args.data_dir)
d.run()
