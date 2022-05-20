import numpy as np
import math
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from scipy.interpolate import griddata


def DE_PSD(data, stft_para):
    '''
    input: data [n*m]          n electrodes, m time points
        stft_para.stftn     frequency domain sampling rate
        stft_para.fStart    start frequency of each frequency band
        stft_para.fEnd      end frequency of each frequency band
        stft_para.window    window length of each sample point(seconds)
        stft_para.fs        original frequency
    output:psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''

    # Initialize the parameters
    STFTN = stft_para['stftn']
    fStart = stft_para['fStart']
    fEnd = stft_para['fEnd']
    fs = stft_para['fs']
    window = stft_para['window']

    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(stft_para['fStart'])):
        fStartNum[i] = int(fStart[i]/fs*STFTN)
        fEndNum[i] = int(fEnd[i]/fs*STFTN)

    n = data.shape[0]
    m = data.shape[1]

    # print(m,n,l)
    psd = np.zeros([n, len(fStart)])
    de = np.zeros([n, len(fStart)])
    # Hanning window
    Hlength = window*fs
    # Hwindow=hanning(Hlength);
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1))
                        for n in range(1, Hlength+1)])

    WindowPoints = fs*window
    dataNow = data[0:n]
    for j in range(0, n):
        temp = dataNow[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0: int(STFTN/2)])
        for p in range(0, len(fStart)):
            E = 0
            E_log = 0
            for p0 in range(fStartNum[p]-1, fEndNum[p]):
                E = E + magFFTdata[p0] * magFFTdata[p0]
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            psd[j][p] = E
            de[j][p] = math.log(100*E, 2)

    return psd, de


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def grid_data(data):
    grid_x, grid_y = np.mgrid[0:4:16j, 0:4:16j]
    points = []
    for i in range(5):
        for j in range(5):
            points.append([i, j])
    values = []
    for x in points:
        values.append(data[x[0]][x[1]][0])
    points = np.array(points)
    values = np.array(values)
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    return grid_z

def norm(pxx):
    mean = pxx.mean(axis=-2)
    std = pxx.std(axis=-2)
    for i in range(pxx.shape[0]):
        for j in range(pxx.shape[1]):
            for k in range(pxx.shape[2]):
                pxx[i][j][k] -= mean[i][k]
    for i in range(pxx.shape[0]):
        for j in range(pxx.shape[1]):
            for k in range(pxx.shape[2]):
                pxx[i][j][k] /= std[i][k]
    return pxx

def convert_heat(pxx):
    heatmap = np.zeros([pxx.shape[0], pxx.shape[2], 5, 5, 1])
    for ep in range(pxx.shape[0]):
        for hz in range(pxx.shape[2]):

            heatmap[ep][hz][0][1][0] = pxx[ep][9][hz]
            heatmap[ep][hz][0][3][0] = pxx[ep][16][hz]

            heatmap[ep][hz][1][0][0] = pxx[ep][8][hz]
            heatmap[ep][hz][1][1][0] = pxx[ep][14][hz]
            heatmap[ep][hz][1][2][0] = pxx[ep][12][hz]
            heatmap[ep][hz][1][3][0] = pxx[ep][11][hz]
            heatmap[ep][hz][1][4][0] = pxx[ep][0][hz]

            heatmap[ep][hz][2][0][0] = pxx[ep][19][hz]
            heatmap[ep][hz][2][1][0] = pxx[ep][7][hz]
            heatmap[ep][hz][2][2][0] = pxx[ep][5][hz]
            heatmap[ep][hz][2][3][0] = pxx[ep][3][hz]
            heatmap[ep][hz][2][4][0] = pxx[ep][13][hz]

            heatmap[ep][hz][3][0][0] = pxx[ep][4][hz]
            heatmap[ep][hz][3][1][0] = pxx[ep][2][hz]
            heatmap[ep][hz][3][2][0] = pxx[ep][17][hz]
            heatmap[ep][hz][3][3][0] = pxx[ep][6][hz]
            heatmap[ep][hz][3][4][0] = pxx[ep][10][hz]

            heatmap[ep][hz][4][1][0] = pxx[ep][18][hz]
            heatmap[ep][hz][4][2][0] = pxx[ep][15][hz]
            heatmap[ep][hz][4][3][0] = pxx[ep][1][hz]
    return heatmap

# Add context to the origin data and label
def AddContext(x, add_context=False):
    '''
    Input:
        x: A tensor whose first axis is number of sample -> (samples, channel, length)
    Output:
        (n_sample, 3, n_channels, n_times)
    '''
    if add_context:
        samples, channel, length = x.shape
        x = x[:, np.newaxis, :, :]
        ContextData = []
        for cur_epoch in range(1, x.shape[0] - 1):
            cur_epoch_data = x[cur_epoch]
            former_epoch = x[cur_epoch - 1]
            latter_epoch = x[cur_epoch + 1]

            temporal_epoch = np.concatenate([former_epoch, cur_epoch_data, latter_epoch], axis=0)
            ContextData.append(temporal_epoch)
        ContextData = np.array(ContextData).swapaxes(1, 2)
        ContextData = ContextData.reshape([samples - 2, channel, 3 * length])
    else:
        x = x[1: x.shape[0] - 1]
        ContextData = x
    return ContextData