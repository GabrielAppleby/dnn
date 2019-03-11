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

    def __init__(self, input_shape, filter_shapes, pooling_schedule,
                 fc_hidden_shape, use_dropout=False, use_bn=False):
        """
        A suggested interface. You can choose to use a different interface and
        make changes to the notebook.

        Model initialization.

        Inputs:
        - input_size: The dimension D of the input data.
        - output_size: The number of classes C.
        - filter_size: sizes of convolutional filters
        - pooling_schedule: positions of pooling layers
        - fc_hidden_size: sizes of hidden layers
        - use_dropout: whether use dropout layers. Dropout rates will be
        specified in training
        - use_bn: whether to use batch normalization

        Return:
        """
        tf.reset_default_graph()

        # Keep options for later
        self.pool_sched = pooling_schedule
        self.use_dropout = use_dropout
        self.use_bn = use_bn

        # For visualization later
        self.ff_layer = None

        with tf.name_scope("create_placeholders"):
            self.input = tf.placeholder(
                tf.float32, shape=input_shape)
            self.labels = tf.placeholder(tf.int32, None)
            self.reg_weight = tf.placeholder(dtype=tf.float32, shape=[])
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
            self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
            self.training_mode = tf.placeholder(dtype=tf.bool, shape=())

        with tf.name_scope("create_learnables"):
            with tf.name_scope("filters"):
                self.filters = []
                self.filter_biases = []

                for counter, filter_size in enumerate(filter_shapes):
                    self.filters.append(
                        tf.get_variable("filter" + str(counter),
                                        filter_size,
                                        dtype=tf.float32))
                    self.filter_biases.append(
                        tf.get_variable("filter_bias" + str(counter),
                                        filter_size[-1],
                                        dtype=tf.float32))
            with tf.name_scope("fully_connected"):
                self.fc_weights = tf.get_variable(
                    "fc_weights", fc_hidden_shape, dtype=tf.float32)
                self.fc_biases = tf.get_variable(
                    "fc_biases", fc_hidden_shape[-1])

        scores = self.compute_scores(self.input)
        self.objective_op = self.compute_objective(scores, self.labels)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.objective_op)
        # predict operation
        self.pred_op = tf.argmax(scores, axis=-1)

        if self.use_bn:
            self.bn_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            self.bn_update_op = []

        # maintain a session for the entire model
        self.session = tf.Session()

    def regularizer(self):
        """
        Calculate the regularization term
        Input:
        Return:
            the regularization term
        """
        with tf.name_scope("regularization"):
            reg = np.float32(0.0)
            for f in self.filters:
                reg = reg + self.reg_weight * tf.reduce_sum(tf.square(f))

            reg = reg + self.reg_weight * tf.reduce_sum(tf.square(
                self.fc_weights))

        return reg

    def compute_scores(self, x):
        """

        Compute the loss and gradients for a two layer fully connected neural
        network. Implement this function in tensorflow

        Inputs:
        - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.

        Returns:
        - scores: a tensor of shape (N, C) where scores[i, c] is the score for
                  class c on input X[i].

        """
        with tf.name_scope("Compute_scores"):
            hidden = x
            for index in range(len(self.filters)):
                hidden = tf.nn.conv2d(
                    hidden, self.filters[index], [1, 1, 1, 1], "SAME")
                hidden = tf.nn.bias_add(hidden, self.filter_biases[index])
                hidden = tf.nn.relu(hidden)

                # Here I deviate from what was suggested in the assignment
                # and perform dropout, and batch normalization after the
                # nonlinearity as is done in keras.
                if self.use_dropout:
                    hidden = tf.nn.dropout(hidden, self.keep_prob)

                if self.use_bn:
                    hidden = tf.layers.batch_normalization(
                        hidden, training=self.training_mode)

                if index in self.pool_sched:
                    hidden = tf.nn.max_pool(
                        hidden, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
            hidden = tf.reshape(
                hidden, shape=[-1, self.fc_weights.shape[0]])
            hidden = tf.matmul(hidden, self.fc_weights)
            scores = tf.nn.bias_add(hidden, self.fc_biases)

        return scores

    def compute_objective(self, scores, y):
        """
        Compute the training objective of the neural network.

        Inputs:
        - scores: A numpy array of shape (N, C). C scores for each instance. C is the number of classes
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - reg: a np.float32 scalar

        Returns:
        - objective: a tensorflow scalar. the training objective, which is the sum of
                     losses and the regularization term
        """
        with tf.name_scope("Compute_objective"):
            softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=scores, labels=y)
            softmax_reduced = tf.reduce_sum(softmax)
            objective = tf.add(
                softmax_reduced, self.regularizer())
        return objective

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, keep_prob=1.0,
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
        - keep_prob: probability of keeping values when using dropout
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        with self.session as sess:
            sess.run(tf.global_variables_initializer())

            objective_history = []
            train_acc_history = []
            val_acc_history = []
            for it in range(num_iters):

                b0 = (it * batch_size) % num_train
                batch = range(b0, min(b0 + batch_size, num_train))

                X_batch = X[batch]
                y_batch = y[batch]

                X = X.astype(np.float32)

                feed_dict = {self.input: X_batch,
                             self.labels: y_batch,
                             self.reg_weight: reg,
                             self.training_mode: True,
                             self.learning_rate: learning_rate,
                             self.keep_prob: keep_prob}

                np_objective, ff_layer, _, _ = sess.run([self.objective_op,
                                                         self.filters[0],
                                                         self.train_op,
                                                         self.bn_update_op],
                                                        feed_dict=feed_dict)
                self.ff_layer = ff_layer
                objective_history.append(np_objective)

                if verbose and it % 100 == 0:
                    print('iteration %d / %d: objective %f' % (
                    it, num_iters, np_objective))

                if it % iterations_per_epoch == 0:
                    # Check accuracy
                    train_acc = np.float32(
                        self.predict(X_batch) == y_batch).mean()
                    val_acc = np.float32(self.predict(X_val) == y_val).mean()
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)

            return {
                'objective_history': objective_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }

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

        np_y_pred = self.session.run(
            self.pred_op, feed_dict={self.input: X, self.training_mode: False})
        return np_y_pred
