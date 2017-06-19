import time
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from caffe_classes import class_names
from alexnet import AlexNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# placeholders for features -- dimension are taken from AlexNet Architecture Paper
x = tf.placeholder(tf.float32, (None, 227, 227, 3))
# By keeping `feature_extract` set to `False`
# we indicate to keep the 1000 class final layer
# originally used to train on ImageNet.
probs = AlexNet(x, feature_extract=False)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Read Images
_im1 = imread("poodle.png")
im1 = (_im1[:, :, :3]).astype(np.float32)
im1 = im1 - np.mean(im1)
_im2 = imread("weasel.png")
im2 = (_im2[:, :, :3]).astype(np.float32)
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
cv2.imshow("image 0", _im2)
cv2.imshow("image 1: ", _im1)
cv2.imshow("Image 0 mean", im1)
cv2.imshow("Image 1 mean", im2)
cv2.waitKey()
