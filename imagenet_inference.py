from helper import read_images, print_output
from alexnet import AlexNet
import tensorflow as tf

im1, im2 = read_images()
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
softmax_probs = AlexNet(x, feature_extract=False)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(softmax_probs, feed_dict={x: [im1, im2]})
    print_output(output)
