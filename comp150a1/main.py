# A bit of setup

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from implementations.b_neural_net import TwoLayerNet
from data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

X_train = np.float32(X_train)
X_val = np.float32(X_val)
X_test = np.float32(X_test)

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

input_size = 32 * 32 * 3
num_classes = 10

best_net = TwoLayerNet(input_size, 128, num_classes)

# Train the network
stats = best_net.train(X_train, y_train, X_val, y_val,
            num_iters=10000, batch_size=200,
            learning_rate=.00001, learning_rate_decay=0.85,
            reg=0.75, verbose=True)

test_acc = (best_net.predict(X_test) == y_test).mean()

print(test_acc)


# hidden_size_list = [8, 16, 32, 64, 128]
# reg_list = [.25, .5, .75, 1]
# learning_rate_decay_list = [.85, .9, .95, 1]
# num_iterations_list = [100, 1000, 10000]
# learning_rate_list = [1e-3, 1e-5, 1e-7]
#
# best_test_acc = -999999
#
# for learn_rate in learning_rate_list:
#     for num_iterations in num_iterations_list:
#         for hidden_size in hidden_size_list:
#             for reg in reg_list:
#                 for decay in learning_rate_decay_list:
#                     net = TwoLayerNet(input_size, hidden_size, num_classes)
#
#                     # Train the network
#                     stats = net.train(X_train, y_train, X_val, y_val,
#                                 num_iters=num_iterations, batch_size=200,
#                                 learning_rate=1e-5, learning_rate_decay=decay,
#                                 reg=reg, verbose=True)
#
#                     test_acc = (net.predict(X_test) == y_test).mean()
#                     if test_acc > best_test_acc:
#                         # Predict on the validation set
#                         print(X_val.shape)
#                         val_acc = np.float32(
#                             np.equal(net.predict(X_val), y_val)).mean()
#                         print('Validation accuracy: ', val_acc)
#                         print('Test accuracy: ', test_acc)
#                         print("Using: " + str(num_iterations) + ". " + str(hidden_size) + ". " + str(reg) + ". " + "lr " + str(learn_rate) + ". " + str(decay))
#                         best_test_acc = test_acc
#                         best_net = net
#                     tf.reset_default_graph()
