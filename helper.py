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


def transfer_learning(previous_layer, n_classes):
    import tensorflow as tf
    # get the shape of layer 7
    shape = (previous_layer.get_shape().as_list()[-1], n_classes)  # use this shape for the weight matrix
    # defined mean and stddev
    mu, stddev = 0, 0.1
    # weight and bias for output layer
    fc8W = tf.Variable(tf.random_normal(shape=shape, mean=mu, stddev=stddev))
    fc8b = tf.Variable(tf.random_normal(shape=[shape[1]], mean=mu, stddev=stddev))
    # find activation function
    # logits = tf.add(tf.matmul(last_layer, fc8W), fc8b)
    return tf.nn.xw_plus_b(previous_layer, fc8W, fc8b), fc8W, fc8b


def print_output(output):
    from caffe_classes import class_names
    import numpy as np

    for input_im_ind in range(output.shape[0]):
        inds = np.argsort(output)[input_im_ind, :]
        print("Image", input_im_ind)
        for i in range(5):
            print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))


def read_images():
    from scipy.misc import imread
    import numpy as np
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    im1 = imread("poodle.png")
    im1 = (im1[:, :, :3]).astype(np.float32)
    im1 = im1 - np.mean(im1)

    im2 = imread("weasel.png")
    im2 = (im2[:, :, :3]).astype(np.float32)
    im2 = im2 - np.mean(im2)
    return im1, im2
