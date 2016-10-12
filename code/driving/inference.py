"""self-driving part"""

import argparse
from io import BytesIO
import cv2
from gopigo import *
import numpy as np
import os
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
parser.add_argument('--imagefile', type=str,
                    help='Image file on which to run inference on.')
parser.add_argument('--nodrive', action='store_true',
                    help='Enable this to debug without driving.')
args = parser.parse_args()

DEFAULT_SPEED = 50

LABELS = {'left': 0, 'right': 1, 'straight': 2}

# Some basic args checking
if args.resize > args.resolution or args.resize <= 0:
    raise Exception("invalid size for resizing image")

def setup_cam():
    import picamera
    cam = picamera.PiCamera()
    cam.resolution = (args.resolution, args.resolution)
    sleep(1)
    return cam

def _scale_to_zero_one(img):
    """convert pixel values from [0, 255] -> [0.0, 1.0]"""
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        return np.multiply(img, 1.0 / 255.0)
    else:
        print("image values already seem to be float")
        return img

def _postprocess(img):
    """common postprocessing, whether img is captured or from file"""
    img = _scale_to_zero_one(img)
    img = img.reshape(1, -1)  # to avoid a scikit-learn deprecation warning later
    return img

def read_image(path):
    if not os.path.isfile(path):
        raise Exception("image file does not exist")
    img = cv2.imread(path)
    if img.shape[:2] != (args.resize, args.resize):
        img = cv2.resize(img, (args.resize, args.resize),
                         interpolation=cv2.INTER_AREA)
    return img.flatten()

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
    return y[0]

def drive():
    if not args.nodrive:
        fwd()
        set_speed(DEFAULT_SPEED)
    while True:
        img = _postprocess(capture())
        print img
        y = predict(model, img)
        if y == LABELS['left']:
            print "left"
            if not args.nodrive:
                set_left_speed(0)
                set_right_speed(DEFAULT_SPEED)
        elif y == LABELS['right']:
            print "right"
            if not args.nodrive:
                set_right_speed(0)
                set_left_speed(DEFAULT_SPEED)
        elif y == LABELS['straight']:
            print "straight"
            if not args.nodrive:
                fwd()
                set_speed(DEFAULT_SPEED)

if not args.imagefile:
    cam = setup_cam()

model = load_model()
print("done loading model")

if args.imagefile:
    img = read_image(args.imagefile)
    img = _postprocess(img)
    print img
    print("start predict")
    predict(model, img)
else:
    drive()

