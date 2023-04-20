import numpy as np
from scipy import signal

# input: 
# data(type:numpy array)(shape:time * 2)
# model(sklearn model or pytorch model)
# flatten(type: bool)(whether to flatten the input as 200 or use 100*2 as the model input)

# output: 
# probanility_map(number of split, 12)
def stroke_probability_map_1(data, model, flatten, model_type):
    split_list = [125, 200, 400]
    probability_map = np.zeros((16, 12*len(split_list)))
    N = data.shape[0]
    count = 0
    segments = 16
    for split_idx in range(len(split_list)):
        step = int(np.floor((N-split_list[split_idx])/(segments-1)))
        data_cur = np.zeros((segments, split_list[split_idx], 2))
        for s_idx in range(segments):
            data_cur[s_idx] = data[(s_idx*step):(s_idx*step+split_list[split_idx]),:]
        data_cur = signal.resample(data_cur, 100, axis=1)
        if flatten:
            data_cur = data_cur.reshape((segments, -1))

        if model_type == 0:
            probability_map[:, (split_idx*12):(split_idx*12+12)] = model.predict_proba(data_cur)
        else:
            probability_map[:, (split_idx*12):(split_idx*12+12)] = model(data_cur)
            
    return probability_map