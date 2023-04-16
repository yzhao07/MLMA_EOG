import numpy as np
from scipy import signal

# input: 
# data(type:numpy array)(shape:time * 2)
# model(sklearn model or pytorch model)
# flatten(type: bool)(whether to flatten the input as 200 or use 100*2 as the model input)

# output: 
# probanility_map(number of split, 12)
def stroke_probability_map(data, model, flatten):
    split_list = [1, 2, 4, 8, 3, 6, 9]
    probability_map = np.zeros((int(np.sum(split_list)), 12))
    N = data.shape[0]
    count = 0
    for split_idx in range(len(split_list)):
        n_ = int(np.floor(N/split_list[split_idx]))
        for i in range(split_list[split_idx]):
            data_cur = signal.resample(data[(i*n_):((i+1)*n_), :], 100, axis=0)
            if flatten:
                data_cur = data_cur.reshape((1, -1))
            probability_map[count] = model(data_cur)
            count += 1
            
    return probability_map