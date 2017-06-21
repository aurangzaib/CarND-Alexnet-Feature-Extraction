from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import tensorflow as tf
from helper import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def evaluate(features, labels, sess):
    total_cost = 0
    total_accuracy = 0
    batches = get_batches(features, labels, 128)
    for x_batch, y_batch in batches:
        c, a = sess.run([cost, accuracy], feed_dict={
            x: x_batch,
            y: y_batch,
            learn_rate: 0.001
        })
        # x_batch.shape[0] --> features in a batch
        total_cost += (c * x_batch.shape[0])
        total_accuracy += (a * x_batch.shape[0])
        # feature.shape[0] --> total features
    return total_cost / features.shape[0], total_accuracy / features.shape[0]


# TODO: Load traffic signs data.
x_train, y_train = load_data('train.p')
x_train, y_train = pre_process(x_train, y_train, is_train=True)
# TODO: Split data into training and validation sets.
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.33,
                                                    random_state=42)
# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, [None])
resized = tf.image.resize_images(x, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
# TODO: Add the final layer for traffic sign classification.
x_h, x_channels, nb_classes, n_samples = get_data_summary(x_train, y_train)
one_hot_y = tf.one_hot(y, nb_classes)
print("classes: {}".format(nb_classes))
# TODO: Define loss, training, accuracy operations.
learn_rate = tf.placeholder(tf.float32)
logits, fc8W, fc8b = transfer_learning(fc7, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
cost = tf.reduce_mean(cross_entropy)  # --> loss
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost, var_list=[fc8W, fc8b])
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(1):
        batch = get_batches(x_train, y_train, 128)
        for x_batch, y_batch in batch:
            sess.run(optimizer, feed_dict={
                x: x_batch,
                y: y_batch,
                learn_rate: 0.001
            })
    test_cost, test_accuracy = evaluate(x_test, y_test, sess)
    print("loss: {:2.3f}, accuracy: {:2.3f}%".format(test_cost, test_accuracy * 100))
