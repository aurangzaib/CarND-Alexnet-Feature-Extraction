import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(filename):
    import pickle
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    reduce_data = 3920
    print(filename + " length: {}".format(len(data['features'][:reduce_data])))
    return data['features'][:reduce_data], data['labels'][:reduce_data]


def grayscale(x):
    import cv2 as cv
    import numpy as np
    for index, image in enumerate(x):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        im2 = np.zeros_like(image)
        im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] = gray, gray, gray
        x[index] = im2
    return x


def normalizer(x):
    import numpy as np
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x = (x - x_min) / (x_max - x_min)
    return x


def pre_process(features, labels, is_train=False):
    from sklearn.utils import shuffle
    assert (len(features) == len(labels))
    # features = grayscale(features)
    features = normalizer(features)
    if is_train:
        features, labels = shuffle(features, labels)
    return features, labels


def get_batches(features, labels, _batch_size_):
    from sklearn.utils import shuffle
    import math
    features, labels = shuffle(features, labels)
    total_size, index, batch = len(features), 0, []
    n_batches = int(math.ceil(total_size / _batch_size_)) if _batch_size_ > 0 else 0
    for _i_ in range(n_batches - 1):
        batch.append([features[index:index + _batch_size_],
                      labels[index:index + _batch_size_]])
        index += _batch_size_
    batch.append([features[index:], labels[index:]])
    return batch


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
        total_cost += (c * x_batch.shape[0])
        total_accuracy += (a * x_batch.shape[0])
    return total_cost / features.shape[0], total_accuracy / features.shape[0]


def get_data_summary(feature, label):
    import numpy as np
    # What's the shape of an traffic sign image?
    image_shape = feature[0].shape
    # How many unique classes/labels there are in the dataset.
    unique_classes, n_samples = np.unique(label,
                                          return_index=False,
                                          return_inverse=False,
                                          return_counts=True)
    n_classes = len(unique_classes)
    n_samples = n_samples.tolist()
    return image_shape[0], image_shape[2], n_classes, n_samples


def transfer_learning(last_layer, n_classes):
    # get the shape of layer 7
    shape = (last_layer.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
    # defined mean and stddev
    mu, stddev = 0, 0.1
    # weight and bias for output layer
    fc8W = tf.Variable(tf.random_normal(shape=shape, mean=mu, stddev=stddev))
    fc8b = tf.Variable(tf.random_normal(shape=[shape[1]], mean=mu, stddev=stddev))
    # find activation function
    # logits = tf.add(tf.matmul(last_layer, fc8W), fc8b)
    return tf.nn.xw_plus_b(last_layer, fc8W, fc8b), fc8W, fc8b


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
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost, var_list=[fc8W, fc8b])
correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(one_hot_y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # 10 epochs
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
