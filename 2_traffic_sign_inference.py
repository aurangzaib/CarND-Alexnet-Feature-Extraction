"""
using Alexnet pre-trained network for
inference of traffic signs
"""
from helper import read_images, print_output
from alexnet import AlexNet
import tensorflow as tf

im1, im2 = read_images('construction.jpg', 'stop.jpg')
# images has 32, 32, 3 dimensions
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
# alexnet requires 227,227,3 dimensions
resized = tf.image.resize_images(x, size=[227, 227])
softmax_probs = AlexNet(resized)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(softmax_probs, feed_dict={x: [im1, im2]})
    print_output(output)
