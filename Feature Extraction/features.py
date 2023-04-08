# Some Functions
import numpy as np
from scipy import stats
import librosa
from sklearn.linear_model import LinearRegression

#%% --------------------计算EMG features--------------------
def EMG_features(data, segments_num, function_list=[]):
    # input data: (time, channels)
    # output: (channels, segments_num, features number) 
    if segments_num > data.shape[0]/2:
        print('segments_num is too large')
        return -1
    elif segments_num < 1:
        print('segment_num is too small')
        return -1
    if len(function_list) < 1:
        print('function_list is empty')
        return -1
    
    # segment signal
    segments_num = int(segments_num)
    channel_num = data.shape[1]
    k = np.floor(data.shape[0], segments_num) # segment length
    data_seg = np.zeros((0, k))
    for i in range(data.shape[1]):
        data_seg = np.append(data_seg, data[:(k*segments_num), i].reshape((segments_num, k)), axis=0)
    data_seg = data_seg.T
    
    # calculate features of each segment
    features = np.zeros((data_seg.shape[1], 0))
    for function_idx in range(len(function_list)):
        if len(function_list[function_idx]) == 1:
            x = function_list[function_idx][0](data_seg)
            features = np.append(features, x, axis=1)
        elif len(function_list[function_idx]) == 2:
            x = function_list[function_idx][0](data_seg, function_list[function_idx][1])
            features = np.append(features, x, axis=1)
    
    features_return = np.zeros((channel_num, segments_num, features.shape[1]))
    for i in range(channel_num):
        features_return[i] = features[int(i*segments_num):int((i+1)*segments_num)]
        
    return features_return
#%% --------------------features functions--------------------
# intput : (time, segments)
# output : (segments, 1) or (segments, n) 

#%% ----------Most basic statistic features----------
# varianve
def var(x):
    return np.var(x, axis=0).reshape((-1, 1))
def mean(x):
    return np.mean(x, axis=0).rehape((-1, 1))
def std(x):
    return np.std(x, axis=0).reshape((-1, 1))
def skew(data):
    return stats.skew(data, axis=0).reshape((-1, 1))
def kurtosis(data):
    return stats.kurtosis(data, axis=0).reshape((-1, 1))
def std_mean(data):
    return (np.std(data, axis=0) / np.mean(data, axis=0)).reshape((-1, 1))


#%% ----------Signal waveform features----------
# mean absolute value
def mav(x):
    return np.mean(np.abs(x), axis=0).reshape((-1, 1))
# modified mean absolute value
# 即给不同的时间点加权
def mav1(x):
    N = x.shape[0]
    weight = np.ones((N, 1)) * 0.5
    begin_n = int(N/4)
    end_n = int(3*N/4)
    weight[begin_n:end_n] = 1
    return np.mean(np.multiply(np.abs(x), weight), axis=0).reshape((-1, 1))
def mav2(x):
    N = x.shape[0]
    weight = np.linspace(start=1, stop=N, num=N, endpoint=True).reshape((N, 1))
    begin_n = int(N/4)
    end_n = int(3*N/4)
    weight[begin_n:end_n] = 1
    weight[:begin_n] = 4*weight[:begin_n]/N
    weight[end_n:] = 4*(N-weight[end_n:])/N
    return np.mean(np.multiply(np.abs(x), weight), axis=0).reshape((-1, 1))

# root mean square
def rms(data):
    return np.sqrt(np.square(data).mean(axis=0)).reshape((-1, 1))

# wavelength
def wl(x):
    return np.mean(np.abs(x[1:] -  x[:-1]), axis=0).reshape((-1, 1))
def wa(x, thr=500):
    return np.mean(np.abs(x[1:] -  x[:-1]) > thr, axis=0).reshape((-1, 1))

# zero crossing: 数据正负号改变的次数
def zc(x):
    threshold = 1e-3
    a = np.multiply(x[:-1, :], x[1:, :]) <= -threshold
    b = np.abs(x[:-1, :] - x[1:, :]) >= threshold
    zc = np.sum((a.astype(int) + b.astype(int)) == 2, axis=0)
    return zc.reshape((-1, 1))

# shape slope chanfe: 数据上升或下降改变的次数
def ssc(x):
    threshold = 1e-3
    slope = np.diff(x, axis=0)
    a = np.multiply(slope[:-1, :], slope[1:, :]) >= threshold
    return np.sum(a, axis=0).reshape((-1, 1))

# auto-regressive (AR) model coefficients
def AR_coef(data, order):
    # input: (times, channels)
    # output: (channels, order+1)
    K = np.floor(data.shape[0]/(order+1))
    ar_coef = np.zeros((data.shape[1], order+1))
    for i in range(data.shape[1]):
        x = data[:int(K*(order+1)), i].reshape((K, order+1))
        reg = LinearRegression().fit(x[:, :-1], x[:, -1])
        ar_coef[i, :-1] = reg.coef_
        ar_coef[i, -1] = reg.intercept_
    return 0


#%% ----------Self similarity features----------
# hurst exponent
def he(data, K=20):
    hm = np.zeros((data.shape[1],))
    # data is decomposed in K subparts of length k with the total number of subparts K = N/k
    N = data.shape[0]
    k = np.floor(N/K)
    for i in range(data.shape[1]):
        x = data[:int(k*K), i].reshape((K, k))
        hm[i] = np.abs(x.mean(axis=1) - np.mean(data[:int(k*K), i])).mean()
        
    return hm.reshape((-1, 1))

# auto-corralation
def auto_cor(data, lag=1):
    var = np.var(data, axis=0)
    temp = np.sum(data[:-lag, :] * data[lag:, :], axis=0)
    return (temp/var/data.shape[0]).reshape((-1, 1))

# The Complexity Coefficient (CC)
def cc(data):
    cc = np.zeros((data.shape[1]))
    data_1 = np.diff(data, axis=0)
    data_2 = np.diff(data_1, axis=0)
    cor_0 = data.var(axis=0)
    cor_1 = data_1.var(axis=0)
    cor_2 = data_2.var(axis=0)

    cc[:] = np.multiply(cor_0, cor_2) / np.square(cor_1)

    return cc.reshape((-1, 1))


#%% ----------MFCC features----------
# MFCC
def mfcc_feature(data,
                 sr,
                 n_mfcc = 10,
                 lifter = 12,
                 n_fft = 500,
                 hop_length = 250,
                 fmin = 10,
                 fmax = 100,
                 pre_em = 0.45):
    # input: (times, channels)
    # output: (channels, (frequency components * time components))
    
    x = librosa.feature.mfcc(y=data[0, 1:] - pre_em * data[0, :-1], \
                                     sr=sr, n_mfcc=n_mfcc, lifter=lifter, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    

    mfcc = np.zeros((data.shape[1], x.shape[0]*x.shape[1]))
    
    for i in range(data.shape[1]):
        x = librosa.feature.mfcc(y=data[0, 1:] - pre_em * data[0, :-1], \
            sr=sr, n_mfcc=n_mfcc, lifter=lifter, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
        mfcc[i] = x.reshape((-1))

    return mfcc

# Mean and standard diviation of each frequency components in MFCC
def mfcc_feature_mean_std(data,
                 sr,
                 n_mfcc = 10,
                 lifter = 12,
                 n_fft = 500,
                 hop_length = 250,
                 fmin = 10,
                 fmax = 100):
    # input: (times, channels)
    # output: (channels, (frequency components * 2))
    pre_em = 0.45
    x = librosa.feature.mfcc(y=data[0, 1:] - pre_em * data[0, :-1], \
                                     sr=sr, n_mfcc=n_mfcc, lifter=lifter, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    

    mfcc = np.zeros((data.shape[1], n_mfcc*2))
    
    for i in range(data.shape[1]):
        x = librosa.feature.mfcc(y=data[0, 1:] - pre_em * data[0, :-1], \
            sr=sr, n_mfcc=n_mfcc, lifter=lifter, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
        mfcc[i, 0] = x.mean(axis=1)
        mfcc[i, 1] = x.std(axis=1)

    return mfcc


#%% ----------正态性检验----------
# 一种检验正态分布的方法
def Shapiro(data):
    test_array = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):
        test_array[i], _ = stats.shapiro(data[:, i])
    return test_array.reshape((-1, 1))

# 一种检验正态分布的方法
def kstest(data):
    test_array = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):
        test_array[i], _ = stats.kstest(data[:, i], 'norm')
    return test_array.reshape((-1, 1))
