from driver import Driver
import os
from sklearn.externals import joblib

class SVC_Driver(Driver):
    def __init__(self, imagefile=None, nodrive=False):
        super(SVC_Driver, self).__init__(imagefile, nodrive)

    def load_model(self, modelfile):
        if not os.path.exists(modelfile):
            raise Exception("model file does not exist")
        self.model = joblib.load(modelfile)


    def predict(self, x):
        """make predictions given model and feature vector x"""
        y = self.model.predict(x)
        print("y")
        print(y)
        return y[0]




#d = SVC_Driver(imagefile="/Users/naturegirl/code/tensor-racer/data/round1/left/1.jpg", nodrive=True)
d = SVC_Driver(nodrive=True)
d.load_model(modelfile="/Users/naturegirl/code/tensor-racer/code/training/SVC/model.pkl")
d.run()
