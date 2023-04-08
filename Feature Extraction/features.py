# Some Functions
import numpy as np
from scipy import stats
import librosa

# !!! 以下函数的输入分两种，一种的输入是time * channels，另一种是channel * times。已标注

#%% ----- Part 1 -----
# intput x: time * channels
# output channels

# mean absolute value
def mav(x):
    return np.mean(np.abs(x), axis=0)

def mav1(x):
    N = x.shape[0]
    weight = np.ones((N, 1)) * 0.5
    begin_n = int(N/4)
    end_n = int(3*N/4)
    weight[begin_n:end_n] = 1
    return np.mean(np.multiply(np.abs(x), weight), axis=0)
# modified mean absolute value
# 即给不同的时间点加权
def mav2(x):
    N = x.shape[0]
    weight = np.linspace(start=1, stop=N, num=N, endpoint=True).reshape((N, 1))
    begin_n = int(N/4)
    end_n = int(3*N/4)
    weight[begin_n:end_n] = 1
    weight[:begin_n] = 4*weight[:begin_n]/N
    weight[end_n:] = 4*(weight[end_n:] - N)/N
    return np.mean(np.multiply(np.abs(x), weight), axis=0)

def mav2_2(x):
    N = x.shape[0]
    weight = np.linspace(start=1, stop=N, num=N, endpoint=True).reshape((N, 1))
    begin_n = int(N/4)
    end_n = int(3*N/4)
    weight[begin_n:end_n] = 1
    weight[:begin_n] = 4*weight[:begin_n]/N
    weight[end_n:] = 4*(N-weight[end_n:])/N
    return np.mean(np.multiply(np.abs(x), weight), axis=0)

def mav2_3(x):
    N = x.shape[0]
    weight = np.linspace(start=1, stop=N, num=N, endpoint=True).reshape((N, 1))
    begin_n = int(N/4)
    weight[begin_n:] = 1
    weight[:begin_n] = 4*weight[:begin_n]/N
    return np.mean(np.multiply(np.abs(x), weight), axis=0)

def mav3(x):
    N = x.shape[0]
    weight = np.ones((N, 1)) * 0.5
    begin_n = int(N/4)
    weight[begin_n:] = 1
    return np.mean(np.multiply(np.abs(x), weight), axis=0)

# wavelength
def wl(x):
    return np.mean(np.abs(x[1:] -  x[:-1]), axis=0)
def wa(x, thr=500):
    return np.mean(np.abs(x[1:] -  x[:-1]) > thr, axis=0)

# varianve
def var(x):
    return np.var(x, axis=0)

# zero crossing: 数据正负号改变的次数
def zc(x):
    threshold = 1e-3
    a = np.multiply(x[:-1, :], x[1:, :]) <= -threshold
    b = np.abs(x[:-1, :] - x[1:, :]) >= threshold
    zc = np.sum((a.astype(int) + b.astype(int)) == 2, axis=0)
    return zc
# shape slope chanfe: 数据上升或下降改变的次数
def ssc(x):
    threshold = 1e-3
    slope = np.diff(x, axis=0)
    a = np.multiply(slope[:-1, :], slope[1:, :]) >= threshold
    return np.sum(a, axis=0)

#%% ----- Part 2 -----
# input: channels * time
# output: channels
# root mean square
def rms(data):
    axis=-1
    return np.sqrt(np.square(data).mean(axis=axis))

# hurst exponent
def he(data, K=20):
    hm = np.zeros((data.shape[0],))
    # data is decomposed in K subparts of length k with the total number of subparts K = N/k
    for i in range(data.shape[0]):
        x = data[i, : ].reshape((K, -1))
        hm[i] = np.abs(x.mean(axis=1) - np.mean(data[i, :])).mean()
        #hm[i] = np.abs(x.mean(axis=-1) - np.mean(data[:, i])).mean()

    return hm

# The Complexity Coefficient (CC)
def cc(data):
    cc = np.zeros((data.shape[0]))
    data_1 = np.diff(data, axis=-1)
    data_2 = np.diff(data_1, axis=-1)
    cor_0 = data.var(axis=-1)
    cor_1 = data_1.var(axis=-1)
    cor_2 = data_2.var(axis=-1)

    cc[:] = np.multiply(cor_0, cor_2) / np.square(cor_1)

    return cc

# MFCC 频域方面的特征
def mfcc_feature(data,
                 sr = 500,
                 n_mfcc = 10,
                 lifter = 12,
                 n_fft = 500,
                 hop_length = 250,
                 fmin = 10,
                 fmax = 100,
                 pre_em = 0.45, 
                 num = 1):
    # input: segments * times * channels
    # output: segments * channels * features number

    f = np.zeros((data.shape[0], 8, n_mfcc*2))
    for i in range(0, data.shape[0], num):
        for channel_idx in range(8):
            # x shape: n_mfcc * times
            x = librosa.feature.mfcc(y=data[i, 1:, channel_idx] - pre_em * data[i, :-1, channel_idx], \
                                     sr=sr, n_mfcc=n_mfcc, lifter=lifter, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
            # f[i, channel_idx] = x.mean(axis=1)
            f[i, channel_idx, :n_mfcc] = x.mean(axis=1)
            f[i, channel_idx, n_mfcc:] = x.std(axis=1)
        
        #if i % 300 == 299:
            #print(i)

    return f

# 自相关
def auto_cor(data, lag=1):
    var = np.var(data, axis=1)
    temp = np.sum(data[:, :-lag] * data[:, lag:], axis=1)
    return temp/var/data.shape[1]

def auto_cor_mad(data, lag=1):
    var = np.square(stats.median_abs_deviation(data, axis=1))
    temp = np.sum(data[:, :-lag] * data[:, lag:], axis=1)
    return temp/var/data.shape[1]

def Gaussion_pearson(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    hist_array = np.zeros((data.shape[0], 10))
    bin_edges_array = np.zeros((data.shape[0], 11))
    for i in range(data.shape[0]):
        hist_array[i], bin_edges_array[i] = np.histogram(data[i], density=True)
        
    dis_array = np.zeros((data.shape[0], 10))

    for i in range(data.shape[0]):
        x = np.append(bin_edges_array[i, :-1].reshape((-1, 1)), bin_edges_array[i, 1:].reshape((-1, 1)), axis=1)
        x = x.mean(axis=1)
        dis_array[i] = stats.norm.pdf(x, mean[i], std[i])
    
    return np.diag(np.corrcoef(data, dis_array))

# 一种检验正态分布的方法
def Shapiro(data):
    test_array = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        test_array[i], _ = stats.shapiro(data[i])
    return test_array

# 一种检验正态分布的方法
def kstest(data):
    test_array = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        test_array[i], _ = stats.kstest(data[i], 'norm')
    return test_array

def skew(data):
    return stats.skew(data, axis=1)

def kurtosis(data):
    return stats.kurtosis(data, axis=1)

def std_mean(data):
    return np.std(data, axis=1) / np.mean(data, axis=1)
