"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with tensorflow instead of
numpy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """

        # store parameters in numpy arrays
        self.params = {}
        self.params['W1'] = tf.Variable(
            std * np.random.randn(input_size, hidden_size), dtype=tf.float32)
        self.params['b1'] = tf.Variable(np.zeros(hidden_size), dtype=tf.float32)
        self.params['W2'] = tf.Variable(
            std * np.random.randn(hidden_size, output_size), dtype=tf.float32)
        self.params['b2'] = tf.Variable(np.zeros(output_size), dtype=tf.float32)

        self.session = None  # will get a tf session at training
        self.x_test = tf.placeholder(dtype=np.float32, shape=[None, None])
        self.predict_node = tf.nn.softmax(self.compute_scores(self.x_test))

    def get_learned_parameters(self):
        """
        Get parameters by running tf variables
        """

        learned_params = dict()

        learned_params['W1'] = self.session.run(self.params['W1'])
        learned_params['b1'] = self.session.run(self.params['b1'])
        learned_params['W2'] = self.session.run(self.params['W2'])
        learned_params['b2'] = self.session.run(self.params['b2'])

        return learned_params

    def softmax_loss(self, scores, y):
        """
        Compute the softmax loss. Implement this function in tensorflow

        Inputs:
        - scores: Input data of shape (N, C), tf tensor. Each scores[i] is a vector
                  containing the scores of instance i for C classes .
        - y: Vector of training labels, tf tensor. y[i] is the label for X[i], and each y[i] is
             an integer in the range 0 <= y[i] < C. This parameter is optional; if it
             is not passed then we only return scores, and if it is passed then we
             instead return the loss and gradients.

        Returns:
        - loss: softmax loss for this batch of training samples.
        """
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=y))

    def compute_scores(self, X):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network. Implement this function in tensorflow

        Inputs:
        - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.

        Returns:
        - scores: a tensor of shape (N, C) where scores[i, c] is the score for
                  class c on input X[i].

        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        layer_one: tf.Tensor = tf.nn.relu((tf.matmul(X, W1) + b1))
        layer_two: tf.Tensor = (tf.matmul(layer_one, W2) + b2)

        return layer_two

    def compute_objective(self, X, y, reg):
        """
        Compute the training objective of the neural network.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - reg: a np.float32 scalar


        Returns:
        - objective: a tensorflow scalar. the training objective, which is the sum of
                     losses and the regularization term
        """
        W1: tf.Variable = self.params['W1']
        W2: tf.Variable = self.params['W2']
        scores: tf.Tensor = self.compute_scores(X)
        loss: tf.Tensor = self.softmax_loss(scores, y)
        objective: tf.Tensor = loss + (reg * (
                tf.math.reduce_sum(tf.square(W1)) +
                tf.math.reduce_sum(tf.square(W2))))

        return objective

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=np.float32(5e-6), num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # Placeholders
        lr = tf.placeholder(tf.float64, shape=[])

        # Shapes and calculations
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Set up data with mini batching
        dataset = tf.data.Dataset.from_tensor_slices((X, y))\
            .batch(batch_size)\
            .repeat()
        iter = dataset.make_one_shot_iterator()
        train_x, train_y = iter.get_next()


        # Calculate objective
        objective: tf.Tensor = self.compute_objective(train_x, train_y, reg)

        # Get the gradient and the update operation
        # TODO does this use SGD??
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(objective)

        # Create session
        session = tf.Session()
        self.session = session

        # Track progress
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        session.run(tf.global_variables_initializer())

        # Actual training loop
        for it in range(num_iters):
            X_batch, y_batch, _, loss = session.run(
                [train_x, train_y, opt, objective],
                feed_dict={lr: learning_rate})  # need to feed in

            loss_history.append(loss)

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X) -> np.array:
        """
        Use the trained weights of this two-layer network to predict labels for
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

        y_pred = np.argmax(self.session.run(
            self.predict_node, feed_dict={self.x_test: X}), axis=1)

        ###########################################################################
        # TODO: Implement this function.                                          #
        ###########################################################################
        #
        # You cannot use tensorflow operations here.
        # Instead, build a computational graph somewhere else and run it here.
        # This function is executed in a for-loop in training
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred
