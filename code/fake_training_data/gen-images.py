#!/usr/bin/python
#
# script to generate training images
# tested on Mac OS X
#
# Installation:
# I already had numpy and python (2.x) installed.
# To install open cv, I ran: "brew install opencv".
# To test whether open cv installed successfully,
# open a python console and run "import cv2"

import argparse
import numpy as np
import cv2
import random

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-s', '--image_size', type=int, default=100,
                   help='size of each image')
parser.add_argument('-n', '--num_images', type=int, default=16,
                    help='total number of images in dataset')

args = parser.parse_args()
print(args.image_size)

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
IMAGE_FILENAME = "train.bmp"
LABELS_FILENAME = "train.txt"

# generate image, and return it together with labels
# labels should be:
#  - direction: left, right, straight
#  - angle: in degrees?
def gen_img():
	n = args.image_size
	line_width = n / 3
	img = np.zeros((n, n, 3), np.uint8)
	theta = np.radians(random_angle())
	print("theta", theta)
	x = n * np.tan(theta) # distance from center
	print("x", x)
	line_end_x = int(n/2 + x)
	print("line end x", line_end_x)
	cv2.line(img, (n/2, n), (line_end_x,0), GREEN, line_width)

	# add border
	# width = 10
	# img=cv2.copyMakeBorder(img, width, width, width, width,
	# 					   borderType = cv2.BORDER_CONSTANT, value = WHITE)

	return img
	# cv2.imshow('image', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

def straight_line():
	img = np.zeros((512,512,3), np.uint8)
	cv2.line(img,(256,512),(256,0),(0,255,0),150)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# sample an angle from (-max, max) degrees
def random_angle(max=45):
	return random.randint(-max, max+1)

def show_img(img):
	cv2.imshow('image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def combine_images(imgs, n_cols, n_rows):
	rows = []
	for i in range(n_rows):
		cols = []
		for j in range(i, i+n_cols):
			cols.append(imgs[j])
		rows.append(np.concatenate((cols), axis=1))
	combined = np.concatenate(rows, axis=0)
	return combined

N = args.num_images # number of images to generate
imgs = [gen_img() for i in range(N)]
combined = combine_images(imgs, 4, 4)
#show_img(combined)
print("write to file")
cv2.imwrite(IMAGE_FILENAME, combined)
