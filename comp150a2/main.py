# As usual, a bit of setup
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from implementations.layers import *
from data_utils import get_CIFAR10_data

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

np.random.seed(15009)

from implementations.conv_net import ConvNet

num_train = 100
X_train = data['X_train'][:num_train].transpose([0, 2, 3, 1])
y_train = data['y_train'][:num_train]
X_val = data['X_val'].transpose([0, 2, 3, 1])
y_val = data['y_val']



model = ConvNet(input_size=[100, 32, 32, 3], 
                output_size=10, 
                filter_size=[4, 3, 3, 5], 
                pooling_schedule=[1, 3], 
                fc_hidden_size=[50])

trace = model.train(X_train, y_train, X_val, y_val,
            learning_rate=1e-3, 
            reg=np.float32(5e-6), 
            num_iters=1000,
            batch_size=20, 
            verbose=True)