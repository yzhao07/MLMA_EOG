# -*- coding: utf-8 -*-
"""preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_HGxPyUOCJ3wnMfPbXl7sqWSz-jsCMGe
"""

# -*- coding: utf-8 -*-
"""preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_HGxPyUOCJ3wnMfPbXl7sqWSz-jsCMGe
"""

# input: data (type: numpy array)(shape: (time, 2))
# output: data (type: numpy array)(shape: (100, 2))
import numpy as np
from scipy import signal

def preprocess(data):
  # Delete silent part
  data = data.astype(float)
  n_size = 50
  n_len = int(data.shape[0]/n_size)
  std_data = np.zeros((n_size, 2))
  for i in range(n_size):
    seg_data_x = data[i*n_len:i*n_len+n_len, 0]
    seg_data_y = data[i*n_len:i*n_len+n_len, 1]
    std_data[i, 0] = np.std(seg_data_x)
    std_data[i, 1] = np.std(seg_data_y)
  pass_threshold = 1
  pass_idx_x = np.where(std_data[:,0] >= pass_threshold)[0]
  pass_idx_y = np.where(std_data[:,1] >= pass_threshold)[0]
  if len(pass_idx_x) == 0:
    start_idx = max(0, pass_idx_y[0] - 1)
    end_idx = min(data.shape[0],pass_idx_y[-1] + 1)
  elif len(pass_idx_y) == 0:
    start_idx = max(0, pass_idx_x[0] - 1)
    end_idx = min(data.shape[0],pass_idx_x[-1] + 1)
  else:
    start_idx = max(0,min(pass_idx_x[0], pass_idx_y[0]) - 1)
    end_idx = min(data.shape[0],max(pass_idx_x[-1], pass_idx_y[-1]) + 1)
  
  # resample to 100 data points
  data = signal.resample(data[start_idx*n_len:end_idx*n_len, :], 100, axis=0)
  # scale
  data = (data - data.min(axis=0, keepdims=True))/(data.max(axis=0, keepdims=True) - data.min(axis=0, keepdims=True))
  return data