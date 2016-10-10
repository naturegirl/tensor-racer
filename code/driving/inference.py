"""self-driving part"""

import argparse
from io import BytesIO
import cv2
import numpy as np
import os
import picamera
from sklearn.externals import joblib
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument('--modelfile', type=str,
                  default='../training/SVC/model.pkl',
                  help='Path to pickle model file')
parser.add_argument('--resolution', type=int, default=100,
                    help='Image resolution')
parser.add_argument('--resize', type=int, default=30,
                    help='Resize images to this size to match training data.')

args = parser.parse_args()

# Camera
stream = BytesIO()
cam = picamera.PiCamera()
cam.resolution = (args.resolution, args.resolution)
sleep(2)

def capture():
    """capture one image and return as 1D numpy array"""
    stream = BytesIO()
    cam.capture(stream, 'jpeg')
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    # "Decode" the image preserving color
    img = cv2.imdecode(data, 1)
    # switch BGR order to RGB order
    img = img[:, :, ::-1]

    # resize image to match training size
    img = cv2.resize(img, (args.resize, args.resize), interpolation=cv2.INTER_AREA) 
    print("done resizing")

#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return img.flatten()

def load_model():
    if not os.path.exists(args.modelfile):
        raise Exception("model file does not exist")
    return joblib.load(args.modelfile)

def predict(model, x):
    """make predictions given model and feature vector x"""
    y = model.predict(x)
    print("y")
    print(y)

model = load_model()
print("done loading model")
img = capture()
print("start predict")
predict(model, img)










