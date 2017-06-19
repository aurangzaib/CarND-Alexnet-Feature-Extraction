"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""
import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# TODO: Resize the images so they can be fed into AlexNet.
resized = tf.image.resize_images(x, [227, 227])
assert resized is not Ellipsis, "resized needs to modify the placeholder image size to (227,227)"
probs = AlexNet(resized)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
_im1 = imread("construction.jpg")
im1 = _im1.astype(np.float32)
im1 = im1 - np.mean(im1)

_im2 = imread("stop.jpg")
im2 = _im2.astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))

import cv2

cv2.imshow("image 0", _im1)
cv2.imshow("image 1: ", _im2)
cv2.imshow("Image 0 mean", im1)
cv2.imshow("Image 1 mean", im2)
cv2.waitKey()
