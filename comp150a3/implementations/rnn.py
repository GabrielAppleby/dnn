"""
In this file, you should implement the forward calculation of the basic RNN
model and the RNN model with GRUs. Please use the provided interface. The
arguments are explained in the documentation of the two functions.
"""

import numpy as np
from scipy.special import expit as sigmoid
import tensorflow as tf


def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.
    inputs:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state
              transformation
        wt_x: shape [input_size, hidden_size], weight matrix for input
              transformation
        bias: shape [hidden_size], bias term
        init_state: shape [batch_size, hidden_size], the initial state of the
                    RNN
        input_data: shape [batch_size, time_steps, input_size], input data of
                    `batch_size` sequences, each of which has length
                    `time_steps` and `input_size` features at each time step.
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the
                 sequence. The output at each time step is exactly the hidden
                 state
        final_state: the final hidden state
    """
    outputs = None
    final_state = None
    h_t_list = []

    for batch, state in zip(input_data, init_state):
        h_t_minus_1 = state
        batch_list = []
        for x_t in batch:
            p_one = np.dot(h_t_minus_1, wt_h)
            p_two = np.dot(x_t, wt_x)
            p_three = np.add(p_one, p_two)
            p_four = np.add(p_three, bias)
            h_t = np.tanh(p_four)
            h_t_minus_1 = h_t
            batch_list.append(h_t)
        h_t_list.append(np.array(batch_list))

    outputs = np.array(h_t_list)
    final_state = outputs[:, -1, :]

    return outputs, final_state


def gru(wtu_h,
        wtu_x,
        biasu,
        wtr_h,
        wtr_x,
        biasr,
        wtc_h,
        wtc_x,
        biasc,
        init_state,
        input_data):
    """
    RNN forward calculation.

    inputs:
        wtu_h: shape [hidden_size, hidden_size], weight matrix for hidden state
               transformation for u gate
        wtu_x: shape [input_size, hidden_size], weight matrix for input
               transformation for u gate
        biasu: shape [hidden_size], bias term for u gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state
               transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input
               transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wtc_h: shape [hidden_size, hidden_size], weight matrix for hidden state
               transformation for candicate hidden state calculation
        wtc_x: shape [input_size, hidden_size], weight matrix for input
               transformation for candicate hidden state calculation
        biasc: shape [hidden_size], bias term for candicate hidden state
               calculation
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of
                    `batch_size` sequences, each of which has length
                    `time_steps` and `input_size` features at each time step.
    outputs:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the
                 sequence. The output at each time step is exactly the hidden
                 state
        final_state: the final hidden state
    """
    outputs = None
    final_state = None

    h_t_list = []

    for batch, state in zip(input_data, init_state):
        h_t_minus_1 = state
        batch_list = []
        for x_t in batch:
            # r_t
            r_t_one = np.dot(h_t_minus_1, wtr_h)
            r_t_two = np.dot(x_t, wtr_x)
            r_t_three = r_t_one + r_t_two + biasr
            r_t = sigmoid(r_t_three)

            # u_t
            u_t_one = np.dot(h_t_minus_1, wtu_h)
            u_t_two = np.dot(x_t, wtu_x)
            u_t_three = u_t_one + u_t_two + biasu
            u_t = sigmoid(u_t_three)

            # c_t
            c_t_one = np.multiply(h_t_minus_1, r_t)
            c_t_two = np.dot(c_t_one, wtc_h)
            c_t_three = np.dot(x_t, wtc_x)
            c_t_four = c_t_two + c_t_three + biasc
            c_t = np.tanh(c_t_four)

            # h_t
            h_t_one = np.multiply(h_t_minus_1, u_t)
            h_t_two = 1 - u_t
            h_t_three = np.multiply(c_t, h_t_two)
            h_t = h_t_one + h_t_three

            h_t_minus_1 = h_t
            batch_list.append(h_t)
        h_t_list.append(np.array(batch_list))

    outputs = np.array(h_t_list)
    final_state = outputs[:, -1, :]
    
    return outputs, final_state

