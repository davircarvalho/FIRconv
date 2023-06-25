'''
FIRconv - 2023
---------------------------------------------------------------------------
Real-time convolution algorithms for Finite Impulse Response (FIR) filters.
---------------------------------------------------------------------------

MIT License

Copyright (c) 2023 Davi Carvalho

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Play around and test the filter modes
'''

# %% Import libs
import numpy as np
from FIRconv import FIRfilter
import matplotlib.pyplot as plt


# %% Load Inputs
fs = 48000
audio_in = np.longdouble(np.random.rand(5 * fs))
N_ch = 1
eps = np.finfo(np.longdouble).resolution
# Impulse response
ir = np.longdouble(np.random.rand(2**10))

audio_in = np.hstack((audio_in, np.zeros_like(ir)))

# %% Initialize FIR filter
buffer_sz = 2**13
methods = ['ola', 'ols', 'upols']
FIRfilt = []
for method in methods:
    FIRfilt.append(FIRfilter(method, buffer_sz, h=ir))

# % Stream audio
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

# Offline convolution
conv_truth = np.convolve(audio_in, ir)


# %% PLot
def normalizeme(x):
    return x / np.max(np.abs(x))


out_ola = np.array(out_ola).flatten()
out_ols = np.array(out_ols).flatten()
out_upols = np.array(out_upols).flatten()

# normalize outputs
conv_truth = normalizeme(conv_truth)
out_ola = normalizeme(out_ola)
out_ols = normalizeme(out_ols)
out_upols = normalizeme(out_upols)

xlim = max(out_upols.shape)

plt.figure()
plt.plot(conv_truth, label='ground truth')
plt.plot(out_ola, label='overlap-add')
plt.plot(out_ols, label='overlap-save')
plt.plot(out_upols, label='uniformly partitioned overlap-save')
plt.xlim([0, xlim])
plt.legend()

plt.figure()
# plt.plot(np.abs(conv_truth[:xlim] - conv_truth[:xlim]), label='ground truth')
plt.plot(np.abs(conv_truth[:xlim] - out_ola[:xlim]), label='Overlap-Add', color='k')
plt.plot(np.abs(conv_truth[:xlim] - out_ols[:xlim]), label='Overlap-Save', alpha=0.45)
plt.plot(np.abs(conv_truth[:xlim] - out_upols[:xlim]), label='Uniformly Partitioned Overlap-Save', alpha=0.45)
plt.axhline(eps, linestyle='dashed', linewidth=2, color='k', label='Numerical resolution')
plt.ylabel('Error')
plt.xlabel('Time (samples)')
plt.xlim([0, xlim])
plt.title('')
plt.legend()


# %%
