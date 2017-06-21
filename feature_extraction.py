import time
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.misc import imread
from alexnet import AlexNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def transfer_learning(previous_layer, nb_classes):
    # get the shape of layer 7
    shape = (previous_layer.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
    # defined mean and stddev
    mu, stddev = 0, 0.1
    # weight and bias for output layer
    fc8W = tf.Variable(tf.random_normal(shape=shape, mean=mu, stddev=stddev))
    fc8b = tf.Variable(tf.random_normal(shape=[shape[1]], mean=mu, stddev=stddev))
    # find activation function
    logits = tf.nn.xw_plus_b(previous_layer, fc8W, fc8b)
    # find softmax probabilities
    return tf.nn.softmax(logits)

sign_names = pd.read_csv('signnames.csv')
nb_classes = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))

# NOTE: By setting `feature_extract` to `True` we return
# the second to last layer.
fc7 = AlexNet(resized, feature_extract=True)
# now we need to implement the output layer
# ourselves as we want to do transfer learning
# only final layer is changed
probs = transfer_learning(fc7, nb_classes)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
print("Image", input_im_ind)
for i in range(5):
    print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
print()

print("Time: %.3f seconds" % (time.time() - t))
