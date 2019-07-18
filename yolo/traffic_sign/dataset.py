import csv
import json
import pickle
import time

from cloudlog import CloudLog
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import exposure
import warnings
import numpy as np
from traffic_sign.utils import print_progress, get_time_hhmmss
from pandas.io.parsers import read_csv
from nolearn.lasagne import BatchIterator
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from collections import namedtuple
import random
import cv2

num_classes = 43

Parameters = namedtuple('Parameters', [
    # Data parameters
    'num_classes', 'image_size',
    # Training parameters
    'batch_size', 'max_epochs', 'log_epoch', 'print_epoch',
    # Optimisations
    'learning_rate_decay', 'learning_rate',
    'l2_reg_enabled', 'l2_lambda',
    'early_stopping_enabled', 'early_stopping_patience',
    'resume_training',
    # Layers architecture
    'conv1_k', 'conv1_d', 'conv1_p',
    'conv2_k', 'conv2_d', 'conv2_p',
    'conv3_k', 'conv3_d', 'conv3_p',
    'fc4_size', 'fc4_p'
])


def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            image = cv2.imread(prefix + row[0], cv2.IMREAD_COLOR)
            image = cv2.resize(image, (32, 32))
            images.append(image)
            labels.append(int(row[7]))  # the 8th column is the label
        gtFile.close()
    labels = np.array(labels)
    images = np.stack(images)
    return images, labels


class AugmentedSignsBatchIterator(BatchIterator):
    """
    Iterates over dataset in batches.
    Allows images augmentation by randomly rotating, applying projection,
    adjusting gamma, blurring, adding noize and flipping horizontally.
    """

    def __init__(self, batch_size, shuffle=False, seed=42, p=0.5, intensity=0.5):
        """
        Initialises an instance with usual iterating settings, as well as data augmentation coverage
        and augmentation intensity.

        Parameters
        ----------
        batch_size:
                    Size of the iteration batch.
        shuffle   :
                    Flag indicating if we need to shuffle the data.
        seed      :
                    Random seed.
        p         :
                    Probability of augmenting a single example, should be in a range of [0, 1] .
                    Defines data augmentation coverage.
        intensity :
                    Augmentation intensity, should be in a [0, 1] range.

        Returns
        -------
        New batch iterator instance.
        """
        super(AugmentedSignsBatchIterator, self).__init__(batch_size, shuffle, seed)
        self.p = p
        self.intensity = intensity

    def transform(self, Xb, yb):
        """
        Applies a pipeline of randomised transformations for data augmentation.
        """
        Xb, yb = super(AugmentedSignsBatchIterator, self).transform(
            Xb if yb is None else Xb.copy(),
            yb
        )

        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]

            Xb = self.rotate(Xb, batch_size)
            Xb = self.apply_projection_transform(Xb, batch_size, image_size)

        return Xb, yb

    def rotate(self, Xb, batch_size):
        """
        Applies random rotation in a defined degrees range to a random subset of images.
        Range itself is subject to scaling depending on augmentation intensity.
        """
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            delta = 30. * self.intensity  # scale by self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode='edge')
        return Xb

    def apply_projection_transform(self, Xb, batch_size, image_size):
        """
        Applies projection transform to a random subset of images. Projection margins are randomised in a range
        depending on the size of the image. Range itself is subject to scaling depending on augmentation intensity.
        """
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace=False):
            tl_top = random.uniform(-d, d)  # Top left corner, top margin
            tl_left = random.uniform(-d, d)  # Top left corner, left margin
            bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
            bl_left = random.uniform(-d, d)  # Bottom left corner, left margin
            tr_top = random.uniform(-d, d)  # Top right corner, top margin
            tr_right = random.uniform(-d, d)  # Top right corner, right margin
            br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
            br_right = random.uniform(-d, d)  # Bottom right corner, right margin

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
            Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order=1, mode='edge')

        return Xb


def preprocess_dataset(X, y=None):
    """
    Performs feature scaling, one-hot encoding of labels and shuffles the data if labels are provided.
    Assumes original dataset is sorted by labels.

    Parameters
    ----------
    X                : ndarray
                       Dataset array containing feature examples.
    y                : ndarray, optional, defaults to `None`
                       Dataset labels in index form.
    Returns
    -------
    A tuple of X and y.
    """
    print("Preprocessing dataset with {} examples:".format(X.shape[0]))

    # Convert to grayscale, e.g. single channel Y
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    # Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)

    for i in range(X.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[i] = exposure.equalize_adapthist(X[i])
        print_progress(i + 1, X.shape[0])

    if y is not None:
        # Convert to one-hot encoding. Convert back with `y = y.nonzero()[1]`
        y = np.eye(num_classes)[y]
        X, y = shuffle(X, y)

    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,))
    return X, y


def class_name(one_hot):
    return signnames[one_hot.nonzero()[0][0]]


def load_pickled_data(file, columns):
    """
    Loads pickled training and test data.

    Parameters
    ----------
    file    :
              Name of the pickle file.
    columns : list of strings
              List of columns in pickled data we're interested in.

    Returns
    -------
    A tuple of datasets for given columns.
    """

    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))


def load_and_process_data(pickled_data_file):
    """
    Loads pickled data and preprocesses images and labels by scaling features,
    shuffling the data and applying one-hot encoding to labels.

    Parameters
    ----------
    pickled_data_file  : string
                         Pickled data file name.
    Returns
    -------
    A tuple of X and y containing preloaded and preprocessed features and labels respectively.
    """
    X, y = load_pickled_data(pickled_data_file, columns=['features', 'labels'])
    X, y = preprocess_dataset(X, y)
    return (X, y)


def flip_extend(X, y):
    """
    Extends existing images dataset by flipping images of some classes. As some images would still belong
    to same class after flipping we extend such classes with flipped images. Images of other would toggle
    between two classes when flipped, so for those we extend existing datasets as well.

    Parameters
    ----------
    X       : ndarray
              Dataset array containing feature examples.
    y       : ndarray, optional, defaults to `None`
              Dataset labels in index form.

    Returns
    -------
    A tuple of X and y.
    """
    # Classes of signs that, when flipped horizontally, should still be classified as the same class
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38],
    ])
    num_classes = 43

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)

    for c in range(num_classes):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis=0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis=0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

    return (X_extended, y_extended)


def extend_balancing_classes(X, y, aug_intensity=0.5, counts=None):
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    Number of generated examples for each class may be provided in `counts`.

    Parameters
    ----------
    X             : ndarray
                    Dataset array containing feature examples.
    y             : ndarray, optional, defaults to `None`
                    Dataset labels in index form.
    aug_intensity :
                    Intensity of augmentation, must be in [0, 1] range.
    counts        :
                    Number of elements for each class.

    Returns
    -------
    A tuple of X and y.
    """
    num_classes = 43

    _, class_counts = np.unique(y, return_counts=True)
    max_c = max(class_counts)
    total = max_c * num_classes if counts is None else np.sum(counts)

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=np.float32)
    y_extended = np.empty([0], dtype=y.dtype)
    print("Extending dataset using augmented data (intensity = {}):".format(aug_intensity))

    for c, c_count in zip(range(num_classes), class_counts):
        # How many examples should there be eventually for this class:
        max_c = max_c if counts is None else counts[c]
        # First copy existing data for this class
        X_source = (X[y == c] / 255.).astype(np.float32)
        y_source = y[y == c]
        X_extended = np.append(X_extended, X_source, axis=0)
        for i in range((max_c // c_count) - 1):
            batch_iterator = AugmentedSignsBatchIterator(batch_size=min(X_source.shape[0], 200), p=1.0,
                                                         intensity=aug_intensity)
            for x_batch, _ in batch_iterator(X_source, y_source):
                X_extended = np.append(X_extended, x_batch, axis=0)
                print_progress(X_extended.shape[0], total)
        batch_iterator = AugmentedSignsBatchIterator(batch_size=min(max_c % c_count, 200), p=1.0,
                                                     intensity=aug_intensity)
        for x_batch, _ in batch_iterator(X_source, y_source):
            X_extended = np.append(X_extended, x_batch, axis=0)
            print_progress(X_extended.shape[0], total)
            break
        # Fill labels for added images set to current class.
        added = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((added), c, dtype=int))

    return ((X_extended * 255.).astype(np.uint8), y_extended)


class Paths(object):
    """
    Provides easy access to common paths we use for persisting
    the data associated with model training.
    """

    def __init__(self, params):
        """
        Initialises a new `Paths` instance and creates corresponding folders if needed.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.
        """
        self.model_name = self.get_model_name(params)
        self.var_scope = self.get_variables_scope(params)
        self.root_path = os.getcwd() + "/models/" + self.model_name + "/"
        self.model_path = self.get_model_path()
        self.train_history_path = self.get_train_history_path()
        self.learning_curves_path = self.get_learning_curves_path()
        os.makedirs(self.root_path, exist_ok=True)

    def get_model_name(self, params):
        """
        Generates a model name with some of the crucial model parameters encoded into the name.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.

        Returns
        -------
        Model name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        model_name = "k{}d{}p{}_k{}d{}p{}_k{}d{}p{}_fc{}p{}".format(
            params.conv1_k, params.conv1_d, params.conv1_p,
            params.conv2_k, params.conv2_d, params.conv2_p,
            params.conv3_k, params.conv3_d, params.conv3_p,
            params.fc4_size, params.fc4_p
        )
        model_name += "_lrdec" if params.learning_rate_decay else "_no-lrdec"
        model_name += "_l2" if params.l2_reg_enabled else "_no-l2"
        return model_name

    def get_variables_scope(self, params):
        """
        Generates a model variable scope with some of the crucial model parameters encoded.

        Parameters
        ----------
        params  : Parameters
                  Structure (`namedtuple`) containing model parameters.

        Returns
        -------
        Variables scope name.
        """
        # We will encode model settings in its name: architecture, optimisations applied, etc.
        var_scope = "k{}d{}_k{}d{}_k{}d{}_fc{}_fc0".format(
            params.conv1_k, params.conv1_d,
            params.conv2_k, params.conv2_d,
            params.conv3_k, params.conv3_d,
            params.fc4_size
        )
        return var_scope

    def get_model_path(self):
        """
        Generates path to the model file.

        Returns
        -------
        Model file path.
        """
        return self.root_path + "model.ckpt"

    def get_train_history_path(self):
        """
        Generates path to the train history file.

        Returns
        -------
        Train history file path.
        """
        return self.root_path + "train_history"

    def get_learning_curves_path(self):
        """
        Generates path to the learning curves graph file.

        Returns
        -------
        Learning curves file path.
        """
        return self.root_path + "learning_curves.png"


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, patience=100, minimize=True):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = None

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.restore_path = self.saver.save(self.session, os.getcwd() + "/early_stopping_checkpoint")
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path != None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


class ModelCloudLog(CloudLog):

    def log_parameters(self, params, train_size, valid_size, test_size):
        """
        Logs model parameters to console and appends the same text representation to the log file.

        Parameters
        ----------
        params    : Parameters
                    Structure (`namedtuple`) containing model parameters.
        train_size: int
                    Size of the training dataset.
        valid_size: int
                    Size of the training dataset.
        test_size : int
                    Size of the training dataset.
        """
        if params.resume_training:
            self("=============================================")
            self("============= RESUMING TRAINING =============")
            self("=============================================")

        self("=================== DATA ====================")
        self("            Training set: {} examples".format(train_size))
        self("          Validation set: {} examples".format(valid_size))
        self("             Testing set: {} examples".format(test_size))
        self("              Batch size: {}".format(params.batch_size))

        self("=================== MODEL ===================")
        self("--------------- ARCHITECTURE ----------------")
        self(" %-*s %-*s %-*s %-*s" % (10, "", 10, "Type", 8, "Size", 15, "Dropout (keep p)"))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 1", 10, "{}x{} Conv".format(params.conv1_k, params.conv1_k), 8, str(params.conv1_d), 15,
            str(params.conv1_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 2", 10, "{}x{} Conv".format(params.conv2_k, params.conv2_k), 8, str(params.conv2_d), 15,
            str(params.conv2_p)))
        self(" %-*s %-*s %-*s %-*s" % (
            10, "Layer 3", 10, "{}x{} Conv".format(params.conv3_k, params.conv3_k), 8, str(params.conv3_d), 15,
            str(params.conv3_p)))
        self(" %-*s %-*s %-*s %-*s" % (10, "Layer 4", 10, "FC", 8, str(params.fc4_size), 15, str(params.fc4_p)))
        self("---------------- PARAMETERS -----------------")
        self("     Learning rate decay: " + (
            "Enabled" if params.learning_rate_decay else "Disabled (rate = {})".format(params.learning_rate)))
        self("       L2 Regularization: " + (
            "Enabled (lambda = {})".format(params.l2_lambda) if params.l2_reg_enabled else "Disabled"))
        self("          Early stopping: " + ("Enabled (patience = {})".format(
            params.early_stopping_patience) if params.early_stopping_enabled else "Disabled"))
        self(" Keep training old model: " + ("Enabled" if params.resume_training else "Disabled"))


def fully_connected(input, size):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.0)
                             )
    return tf.matmul(input, weights) + biases


def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))


def conv_relu(input, kernel_size, depth):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable('weights',
                              shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[depth],
                             initializer=tf.constant_initializer(0.0)
                             )
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def pool(input, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(
        input,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='SAME'
    )


def model_pass(input, params, is_training):
    """
    Performs a full model pass.

    Parameters
    ----------
    input         : Tensor
                    NumPy array containing a batch of examples.
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    is_training   : Tensor of type tf.bool
                    Flag indicating if we are training or not (e.g. whether to use dropout).

    Returns
    -------
    Tensor with predicted logits.
    """
    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params.conv2_k, depth=params.conv2_d)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params.conv3_k, depth=params.conv3_d)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params.conv3_p), lambda: pool3)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool1, pool2, pool3], 1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params.num_classes)
    return logits


def plot_curve(axis, params, train_column, valid_column, linewidth=2, train_linestyle="b-", valid_linestyle="g-"):
    """
    Plots a pair of validation and training curves on a single plot.
    """
    model_history = np.load(Paths(params).train_history_path + ".npz")
    train_values = model_history[train_column]
    valid_values = model_history[valid_column]
    epochs = train_values.shape[0]
    x_axis = np.arange(epochs)
    axis.plot(x_axis[train_values > 0], train_values[train_values > 0], train_linestyle, linewidth=linewidth,
              label="train")
    axis.plot(x_axis[valid_values > 0], valid_values[valid_values > 0], valid_linestyle, linewidth=linewidth,
              label="valid")
    return epochs


# Plots history of learning curves for a specific model.
def plot_learning_curves(params):
    """
    Plots learning curves (loss and accuracy on both training and validation sets) for a model identified by a parameters struct.
    """
    curves_figure = plt.figure(figsize=(10, 4))
    axis = curves_figure.add_subplot(1, 2, 1)
    epochs_plotted = plot_curve(axis, params, train_column="train_accuracy_history",
                                valid_column="valid_accuracy_history")

    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(50., 115.)
    plt.xlim(0, epochs_plotted)

    axis = curves_figure.add_subplot(1, 2, 2)
    epochs_plotted = plot_curve(axis, params, train_column="train_loss_history", valid_column="valid_loss_history")

    plt.grid()
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.ylim(0.0001, 10.)
    plt.xlim(0, epochs_plotted)
    plt.yscale("log")


def train_model(params, X_train, y_train, X_valid, y_valid, X_test, y_test, logger_config):
    """
    Performs model training based on provided training dataset
    according to provided parameters, and then evaluates trained
    model with testing dataset.
    Part of the training dataset may be used for validation during
    training if specified in model parameters.

    Parameters
    ----------
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    X_train       :
                    Training dataset.
    y_train       :
                    Training dataset labels.
    X_valid       :
                    Validation dataset.
    y_valid       :
                    Validation dataset labels.
    X_test        :
                    Testing dataset.
    y_test        :
                    Testing dataset labels.
    logger_config :
                    Logger configuration, containing Dropbox and Telegram settings
                    for notifications and cloud logs backup.
    """

    # Initialisation routines: generate variable scope, create logger, note start time.
    paths = Paths(params)
    log = ModelCloudLog(
        os.path.join(paths.root_path, "logs"),
        dropbox_token="123",
        telegram_token="123",
        telegram_chat_id="123"
    )
    start = time.time()
    model_variable_scope = paths.var_scope

    log.log_parameters(params, y_train.shape[0], y_valid.shape[0], y_test.shape[0])

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        tf_x_batch = tf.placeholder(tf.float32, shape=(None, params.image_size[0], params.image_size[1], 1))
        tf_y_batch = tf.placeholder(tf.float32, shape=(None, params.num_classes))
        is_training = tf.placeholder(tf.bool)
        current_epoch = tf.Variable(0, trainable=False)  # count the number of epochs

        # Model parameters.
        if params.learning_rate_decay:
            learning_rate = tf.train.exponential_decay(params.learning_rate, current_epoch,
                                                       decay_steps=params.max_epochs, decay_rate=0.01)
        else:
            learning_rate = params.learning_rate

        # Training computation.
        with tf.variable_scope(model_variable_scope):
            logits = model_pass(tf_x_batch, params, is_training)
            if params.l2_reg_enabled:
                with tf.variable_scope('fc4', reuse=True):
                    l2_loss = tf.nn.l2_loss(tf.get_variable('weights'))
            else:
                l2_loss = 0

        predictions = tf.nn.softmax(logits)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_batch, logits=logits)
        loss = tf.reduce_mean(softmax_cross_entropy) + params.l2_lambda * l2_loss

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(loss)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())

        # A routine for evaluating current model parameters
        def get_accuracy_and_loss_in_batches(X, y):
            p = []
            sce = []
            batch_iterator = BatchIterator(batch_size=128)
            for x_batch, y_batch in batch_iterator(X, y):
                [p_batch, sce_batch] = session.run([predictions, softmax_cross_entropy], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: False
                }
                                                   )
                p.extend(p_batch)
                sce.extend(sce_batch)
            p = np.array(p)
            sce = np.array(sce)
            accuracy = 100.0 * np.sum(np.argmax(p, 1) == np.argmax(y, 1)) / p.shape[0]
            loss = np.mean(sce)
            return (accuracy, loss)

        # If we chose to keep training previously trained model, restore session.
        if params.resume_training:
            try:
                tf.train.Saver().restore(session, paths.model_path)
            except Exception as e:
                log("Failed restoring previously trained model: file does not exist.")
                pass

        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), session, patience=params.early_stopping_patience,
                                       minimize=True)
        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)
        if params.max_epochs > 0:
            log("================= TRAINING ==================")
        else:
            log("================== TESTING ==================")
        log(" Timestamp: " + get_time_hhmmss())
        log.sync()

        for epoch in range(params.max_epochs):
            current_epoch = epoch
            # Train on whole randomised dataset in batches
            batch_iterator = BatchIterator(batch_size=params.batch_size, shuffle=True)
            for x_batch, y_batch in batch_iterator(X_train, y_train):
                session.run([optimizer], feed_dict={
                    tf_x_batch: x_batch,
                    tf_y_batch: y_batch,
                    is_training: True
                }
                            )

            # If another significant epoch ended, we log our losses.
            if (epoch % params.log_epoch == 0):
                # Get validation data predictions and log validation loss:
                valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)

                # Get training data predictions and log training loss:
                train_accuracy, train_loss = get_accuracy_and_loss_in_batches(X_train, y_train)

                if (epoch % params.print_epoch == 0):
                    log("-------------- EPOCH %4d/%d --------------" % (epoch, params.max_epochs))
                    log("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                    log("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                    log("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    log("   Elapsed time: " + get_time_hhmmss(start))
                    log("      Timestamp: " + get_time_hhmmss())
                    log.sync()
                    print("-------------- EPOCH %4d/%d --------------" % (epoch, params.max_epochs))
                    print("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, train_accuracy))
                    print("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, valid_accuracy))
                    print("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    print("   Elapsed time: " + get_time_hhmmss(start))
                    print("      Timestamp: " + get_time_hhmmss())
                    #print.sync()
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if params.early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    log("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        # Evaluate on test dataset.
        test_accuracy, test_loss = get_accuracy_and_loss_in_batches(X_test, y_test)
        valid_accuracy, valid_loss = get_accuracy_and_loss_in_batches(X_valid, y_valid)
        print("=============================================")
        print(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, valid_accuracy))
        print(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, test_accuracy))
        print(" Total time: " + get_time_hhmmss(start))
        print("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(session, paths.model_path)
        print("Model file: " + saved_model_path)
        np.savez(paths.train_history_path, train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history, valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)
        print("Train history file: " + paths.train_history_path)
        log.sync(notify=True, message="Finished training with *%.2f%%* accuracy on the testing set (loss = *%.6f*)." % (
            test_accuracy, test_loss))

        plot_learning_curves(params)
        log.add_plot(notify=True, caption="Learning curves")

        plt.show()

'''
train_balanced_preprocessed_dataset_file = "traffic-signs-data/train_balanced_preprocessed.p"
test_preprocessed_dataset_file = "traffic-signs-data/test_preprocessed.p"
train_extended_dataset_file = "traffic-signs-data/train_extended.p"
train_balanced_dataset_file = "traffic-signs-data/train_balanced.p"
train_dataset_file = "traffic-signs-data/train.p"
test_dataset_file = "traffic-signs-data/test.p"
signnames = read_csv("signnames.csv").values[:, 1]
X_train, y_train = readTrafficSigns('./traffic-signs-data/Final_Training/Images')
pickle.dump({
    "features": X_train,
    "labels": y_train
}, open(train_dataset_file, "wb"))
X_train, y_train = load_pickled_data(train_dataset_file, ['features', 'labels'])
# X_train, y_train = preprocess_dataset(X_train, y_train)
print("Number of training examples in initial dataset =", X_train.shape[0])
_, class_counts = np.unique(y_train, return_counts=True)
X_train, y_train = flip_extend(X_train, y_train)
print("Number of training examples after horizontal flipping =", X_train.shape[0])
X_train, y_train = flip_extend(X_train, y_train)
pickle.dump({
    "features": X_train,
    "labels": y_train
}, open(train_balanced_dataset_file, "wb"))
print("Preprocessed balanced training dataset saved in", train_balanced_dataset_file)
X_train, y_train = load_and_process_data(train_balanced_dataset_file)
pickle.dump({
    "features": X_train,
    "labels": y_train
}, open(train_balanced_preprocessed_dataset_file, "wb"))
print("Preprocessed balanced training dataset saved in", train_balanced_preprocessed_dataset_file)

X_test, y_test = readTrafficSigns('./traffic-signs-data/Online-Test-sort/')
pickle.dump({
    "features": X_test,
    "labels": y_test
}, open(test_dataset_file, "wb"))
X_test, y_test = load_and_process_data(test_dataset_file)
pickle.dump({
    "features": X_test,
    "labels": y_test
}, open(test_preprocessed_dataset_file, "wb"))
print("Preprocessed extended testing dataset saved in", test_preprocessed_dataset_file)
'''
signnames = read_csv("signnames.csv").values[:, 1]
train_extended_preprocessed_dataset_file = "traffic-signs-data/train_extended_preprocessed.p"
train_balanced_preprocessed_dataset_file = "traffic-signs-data/train_balanced_preprocessed.p"
test_preprocessed_dataset_file = "traffic-signs-data/test_preprocessed.p"
parameters = Parameters(
    # Data parameters
    num_classes = 43,
    image_size = (32, 32),
    # Training parameters
    batch_size = 256,
    max_epochs = 10,
    log_epoch = 1,
    print_epoch = 1,
    # Optimisations
    learning_rate_decay = False,
    learning_rate = 0.0001,
    l2_reg_enabled = True,
    l2_lambda = 0.0001,
    early_stopping_enabled = True,
    early_stopping_patience = 100,
    resume_training = True,
    # Layers architecture
    conv1_k = 5, conv1_d = 32, conv1_p = 0.9,
    conv2_k = 5, conv2_d = 64, conv2_p = 0.8,
    conv3_k = 5, conv3_d = 128, conv3_p = 0.7,
    fc4_size = 1024, fc4_p = 0.5
)

X_train, y_train = load_pickled_data(train_balanced_preprocessed_dataset_file, columns = ['features', 'labels'])
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25)
X_test, y_test = load_pickled_data(test_preprocessed_dataset_file, columns = ['features', 'labels'])
train_model(parameters, X_train, y_train, X_valid, y_valid, X_test, y_test, None)