import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data
from implementations.layers import dropout_forward

# If running on Gabe's laptop..
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

from implementations.fc_net import FullyConnectedNet

np.random.seed(15009)
# Try training a very deep net with batchnorm
hidden_dims = [100, 100, 100, 100, 100]

num_train = 1000

X_train = data['X_train'][:num_train]
X_train = np.reshape(X_train, [X_train.shape[0], -1])
y_train = data['y_train'][:num_train]

X_val = data['X_val']
X_val = np.reshape(X_val, [X_val.shape[0], -1])
y_val = data['y_val']

bn_model = FullyConnectedNet(input_size=X_train.shape[1],
                             hidden_size=hidden_dims,
                             output_size=10,
                             centering_data=True,
                             use_dropout=False,
                             use_bn=True)

# use an aggresive learning rate
bn_trace = bn_model.train(X_train, y_train, X_val, y_val,
                          learning_rate=5e-4,
                          reg=np.float32(0.01),
                          keep_prob=0.5,
                          num_iters=800,
                          batch_size=100,
                          verbose=True)  # train the model with batch normalization