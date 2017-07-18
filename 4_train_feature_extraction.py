"""
you are given Alexnet network with pre-trained weights (alexnet.py uses bvlx-alexnet.npy).
use it for transfer learning with feature extraction.
last layer is replaced and weights are retrained.
other weights are frozen.
loss, accuracy and updated weights are found.
"""
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import tensorflow as tf
from helper import *

x_train, y_train = load_data('train.p')
x_train, y_train = pre_process(x_train, y_train, is_train=True)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
h, channels, n_classes, n_samples = get_data_summary(x_train, y_train)

x = tf.placeholder(tf.float32, [None, h, h, channels])
y = tf.placeholder(tf.int32, [None])

y_one_hot = tf.one_hot(y, n_classes)
x_resized = tf.image.resize_images(x, [227, 227])  # 32x32x3 --> 227z227x3

network = AlexNet(x_resized, feature_extract=True)
network = tf.stop_gradient(network)

logits, w, b = implement_feature_extraction(network, n_classes, with_prob=False)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cross_entropy)
optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, var_list=[w, b])
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batches = get_batches(x_train, y_train, batch_size=128)
    for x_batch, y_batch in batches:
        sess.run(optimize, feed_dict={
            x: x_batch, y: y_batch
        })
    test_cost, test_accuracy = evaluate(x_test, y_test, cost, accuracy, x, y, sess)
    print("loss: {}, accuracy: {}".format(test_cost, test_accuracy))
