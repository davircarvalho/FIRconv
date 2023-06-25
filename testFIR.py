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
import librosa as lb
import pyaudio
import numpy as np
from FIRconv import FIRfilter


# %% Load Inputs
def mono2stereo(audio):
    if np.size(audio.shape) < 2:
        audio = np.expand_dims(audio, 0)
        audio = np.append(audio, audio, axis=0)
    return audio.T


# Audio input
audio_in, fs = lb.load('letter.wav', sr=None, mono=False, duration=10, dtype=np.float32)  # binaural input signal
audio_in = mono2stereo(audio_in)
N_ch = audio_in.shape[-1]

# Impulse response
ir, _ = lb.load('narrow.wav', sr=fs, mono=False, dtype=np.float32)  # binaural input signal
ir = mono2stereo(ir)


# %% Initialize FIR filter
buffer_sz = 2048
method = 'upols'
FIRfilt = FIRfilter(method, buffer_sz, h=ir)
# %% Stream audio
# instantiate PyAudio (1)
p = pyaudio.PyAudio()
# open stream (2)
stream = p.open(format=pyaudio.paFloat32,
                channels=N_ch,
                rate=fs,
                output=True,
                frames_per_buffer=buffer_sz)

# play stream (3)
frame_start = 0
frame_end = frame_start + buffer_sz
data_out = np.zeros((buffer_sz, N_ch))

mout = []

while frame_end <= max(audio_in.shape):
    # process data
    data_out = FIRfilt.process(audio_in[frame_start:frame_end, :])

    # output data
    data_out = np.ascontiguousarray(data_out, dtype=np.float32)
    stream.write(data_out, buffer_sz)

    # update reading positions
    frame_start = frame_end
    frame_end = frame_start + buffer_sz

# stop stream (4)
stream.stop_stream()
stream.close()
# close PyAudio (5)
p.terminate()


# %%
