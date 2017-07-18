"""
you are given Alexnet network with pre-trained weights (alexnet.py uses bvlx-alexnet.npy).
use it for transfer learning with feature extraction.
last layer is replaced and weights are retrained.
"""
from helper import read_images, print_output, implement_feature_extraction
from alexnet import AlexNet
import tensorflow as tf

im1, im2 = read_images('construction.jpg', 'stop.jpg')
n_classes = 43
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
# Alexnet requires (227,277,3)
resized = tf.image.resize_images(x, [227, 227])
network = AlexNet(resized, feature_extract=True)
probs = implement_feature_extraction(network, n_classes)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(probs, feed_dict={x: [im1, im2]})
    print_output(output)
