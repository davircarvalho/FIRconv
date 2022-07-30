'''
pyFIR - 2022
--------------------------------------------
Python FIR filters for real-time convolution
--------------------------------------------
Authors: Davi Rocha Carvalho


Play around and test the filter modes
'''

# %% Import libs
import os
import librosa as lb
import pyaudio
import numpy as np
from pyFIR import OLA
from SOFASonix import SOFAFile as SOFA


# %% Load Inputs
def mono2stereo(audio):
    if np.size(audio.shape) < 2:
        audio = np.expand_dims(audio, 0)
        audio = np.append(audio, audio, axis=0)
    return audio


# Audio input
audio_in, fs = lb.load('pink.wav', sr=None, mono=False, duration=30, dtype=np.float32)  # binaural input signal
audio_in = mono2stereo(audio_in).T
N_ch = audio_in.shape[1]

# Impulse response
local = r'D:\Documentos\1 - Work\Individualized_HRTF_Synthesis\Datasets\HUTUBS'
file = os.path.join(local, 'pp1_HRIRs_measured.sofa')
Obj = SOFA.load(file)
ir = Obj.Data_IR


# %% Initialize FIR filter
buffer_sz = 1024
FIRfilt = OLA(B=buffer_sz, h=ir[0, :, :].T, normalize=True)

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

cont = 0
idx_pos = 0
while frame_end <= max(audio_in.shape):
    cont += 1
    if cont % 2 == 0:
        idx_pos += 1
    if idx_pos >= ir.shape[0] - 1:
        idx_pos = 0

    # process data
    data_out = FIRfilt.process(audio_in[frame_start:frame_end, :], ir[idx_pos, :, :].T)

    # output data
    stream.write(data_out.astype(np.float32), buffer_sz)

    # update reading positions
    frame_start = frame_end
    frame_end = frame_start + buffer_sz

# stop stream (4)
stream.stop_stream()
stream.close()
# close PyAudio (5)
p.terminate()

# %%
