import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):

   pre_act = np.tanh(x_t@Wx + h_prev@Wh + b)
   return pre_act
  
