'''
pyFIR - 2022
--------------------------------------------
Python FIR filters for real-time convolution
--------------------------------------------
Authors: Davi Rocha Carvalho


Play around and test the filter modes
'''

# %% Import libs
import librosa as lb
import pyaudio
import numpy as np
from pyFIR import FIRfilter
import matplotlib.pyplot as plt


# %% Load Inputs
fs = 44100
audio_in = np.random.rand(10*fs)
N_ch = 1

# Impulse response
ir = np.random.rand(fs)


# %% Initialize FIR filter
buffer_sz = fs
methods = ['ola', 'ols', 'upols']
FIRfilt = []
for method in methods:
    FIRfilt.append(FIRfilter(method, buffer_sz, h=ir, ref=1))

# %% Stream audio
frame_start = 0
frame_end = frame_start + buffer_sz
data_out = np.zeros((buffer_sz, N_ch))

out_ola = []
out_ols = []
out_upols = []

while frame_end <= max(audio_in.shape):
    # process data
    out_ola.append(FIRfilt[0].process(audio_in[frame_start:frame_end]))
    out_ols.append(FIRfilt[1].process(audio_in[frame_start:frame_end]))
    out_upols.append(FIRfilt[2].process(audio_in[frame_start:frame_end]))

    # update reading positions
    frame_start = frame_end
    frame_end = frame_start + buffer_sz


# %% Offline convolution
def fft_conv(x, h):
    nfft = np.shape(x)[0] + np.shape(h)[0] - 1
    X = np.fft.fft(x, nfft)
    H = np.fft.fft(h, nfft)
    return np.fft.ifft(X * H).real


conv_truth = fft_conv(audio_in, ir)

# %% PLot
out_ola = np.array(out_ola).flatten()
out_ols = np.array(out_ols).flatten()
out_upols = np.array(out_upols).flatten()

xlim = 400000

plt.figure()
plt.plot(conv_truth, label='ground truth')
plt.plot(out_ola, label='overlap-add')
plt.plot(out_ols, label='overlap-save')
plt.plot(out_upols, label='uniformly partitioned overlap-save')
plt.xlim([0, xlim])
plt.legend()

plt.figure()
plt.plot(conv_truth[:xlim]-conv_truth[:xlim], label='ground truth')
plt.plot(conv_truth[:xlim]-out_ola[:xlim], label='overlap-add')
plt.plot(conv_truth[:xlim]-out_ols[:xlim], label='overlap-save')
plt.plot(conv_truth[:xlim]-out_upols[:xlim], label='uniformly partitioned overlap-save')
plt.legend()
plt.xlim([0, xlim])
plt.title('Error')
# %%
