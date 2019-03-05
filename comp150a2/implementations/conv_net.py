"""
Implementation of convolutional neural network. Please implement your own convolutional neural network 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ConvNet(object):
  """
  A convolutional neural network. 
  """

  def __init__(self, input_size, output_size, filter_size, pooling_schedule, fc_hidden_size,  weight_scale=None, centering_data=False, use_dropout=False, use_bn=False):
    """
    A suggested interface. You can choose to use a different interface and make changes to the notebook.

    Model initialization.

    Inputs:
    - input_size: The dimension D of the input data.
    - output_size: The number of classes C.
    - filter_size: sizes of convolutional filters
    - pooling_schedule: positions of pooling layers 
    - fc_hidden_size: sizes of hidden layers of hidden layers 
    - weight_scale: the initialization scale of weights
    - centering_data: Whether centering the data or not
    - use_dropout: whether use dropout layers. Dropout rates will be specified in training
    - use_bn: whether to use batch normalization

    Return: 
    """

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=1.0, keep_prob=1.0, 
            reg=np.float32(5e-6), num_iters=100,
            batch_size=200, verbose=False):
    """
    A suggested interface of training the model.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - keep_prob: probability of keeping values when using dropout
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """

  def predict(self, X):
    """
    Use the trained weights of the neural network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    np_y_pred = None    

    return np_y_pred


