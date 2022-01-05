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


# %% Load Inputs
# Audio input
audio_in, fs = lb.load('letter.wav', sr=None, mono=False, duration=10, dtype=np.float32) # binaural input signal
if np.size(audio_in.shape) < 2:
    audio_in = np.expand_dims(audio_in, 0)
    audio_in = np.append(audio_in, audio_in, axis=0)
audio_in = audio_in *0.1

# Impulse response
ir, _ = lb.load('default.wav', sr=fs, mono=False, dtype = np.float32) # binaural input signal
if np.size(ir.shape) < 2:
    ir = np.expand_dims(ir, 0)
    ir = np.append(ir, audio_in, axis=0)
ir = ir *0.1


# %% Initialize FIR filter
buffer_sz = 4096
# (optional) find optimal size UPOLS sub-filter partitions
partition_size, _ = FIRfilter.optimize_UPOLS_parameters(FIRfilter,N=max(ir.shape), B=buffer_sz)

method = 'UPOLS'
firL = FIRfilter(method, buffer_sz, partition=partition_size)  
firR = FIRfilter(method, buffer_sz, partition=partition_size)


# %% Stream audio 
# instantiate PyAudio (1)
p = pyaudio.PyAudio()
# open stream (2)
stream = p.open(format=pyaudio.paFloat32,
                channels=audio_in.shape[0],
                rate=fs,
                input=True,
                output = True,
                frames_per_buffer=buffer_sz)
# play stream (3)
frame_start = 0
frame_end = frame_start + buffer_sz
data = audio_in[:,frame_start:frame_end]
data_out = data * 0

while frame_end <= max(audio_in.shape):  
    # process data 
    data_out[0,:] = firL.process(data[0,:], ir[0,:]) 
    data_out[1,:] = firR.process(data[1,:], ir[1,:]) 
    out = np.transpose(data_out)
    
    # output data   
    stream.write(out*5, buffer_sz)
      
    # update reading positions
    frame_start = frame_end + 1
    frame_end = frame_start + buffer_sz
    data = audio_in[:,frame_start:frame_end]
    
# stop stream (4)
stream.stop_stream()
stream.close()
# close PyAudio (5)
p.terminate()

