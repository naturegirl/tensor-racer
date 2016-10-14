import argparse
from driver import Driver
import os
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str,
                  default='../training/SVC/model.pkl',
                  help='Path to pickle model file')
parser.add_argument('--imagefile', type=str,
                    help='Image file on which to run inference on.')
args = parser.parse_args()

class SVC_Driver(Driver):
    def __init__(self, modelfile, imagefile=None, nodrive=False):
        super(SVC_Driver, self).__init__(modelfile, imagefile, nodrive)

    def load_model(self, modelfile):
        if not os.path.exists(modelfile):
            raise Exception("model file does not exist")
        return joblib.load(modelfile)


    def predict(self, x):
        """make predictions given model and feature vector x"""
        y = self.model.predict(x)
        print("y")
        print(y)
        return y[0]


d = SVC_Driver(modelfile=args.modelfile, imagefile=args.imagefile, nodrive=True)
d.run()
