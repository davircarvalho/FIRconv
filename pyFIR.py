'''
pyFIR - 2022
------------------------------------------------------------------------------
Python FIR filters for real-time convolution
------------------------------------------------------------------------------

MIT License

Copyright (c) 2022 Davi Carvalho

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
'''

import numpy as np
from numpy.fft import fft, ifft


class FIRfilter():
    '''
        method:  'overlap-save'(default) or 'overlap-add'
        B: defines the block length
        h: optional call, initializes the impulse response to be convolved
    '''
    
    def __init__(self, method="overlap-save", B=512, h=None):    
        self.method = method
        self.B = B
        self.NFFT = None
        self.num_input_blocks = None
        self.y = None
        self.flagIRchanged = True
        self.Ny = None
        self.stored_h = h
        self.left_overs = np.zeros((B,)) # remaining samples from last ifft (OLA)
               
    
    def len(x):
        return max(np.shape(x))
                

    def pad_zeros_to(self, x, new_length):
        output = np.zeros((new_length,))
        output[:x.shape[0]] = x
        return output


    def next_power_of_2(self, n):
        return 1 << (int(np.log2(n - 1)) + 1)


    def fft_conv(self,x, h):
        # Calculate fft
        X = fft(x, self.NFFT)
        if self.flagIRchanged: # store the IR fft
            self.H = fft(h, self.NFFT)
            self.flagIRchanged = False
        # Perform convolution
        # Go bacK to time domain       
        return ifft(X * self.H).real 


    def OLA(self, x, h):
        if self.NFFT is None:
            Nx = max(x.shape)
            Nh = max(h.shape)
            self.NFFT = self.next_power_of_2(Nx + Nh - 1) 
        
        # Fast convolution
        y = self.fft_conv(x, h)

        # Overlap-Add the partial convolution result
        out = y[:self.B] + self.left_overs[:self.B]
        
        self.left_overs = np.delete(self.left_overs, [range(0,self.B)], 0)    
        pad = len(y[self.B+1:])
        self.left_overs = self.pad_zeros_to(self.left_overs,pad) + y[self.B+1:]        
        return out


    def OLS(self, x, h):
        if self.NFFT is None:
            Nx = max(x.shape)
            Nh = max(h.shape)
            self.NFFT = self.next_power_of_2(Nx + Nh - 1) 
            # Input buffer
            self.OLS_buffer = np.zeros(shape=(self.NFFT,))        
        
        # Sliding window of the input
        self.OLS_buffer = np.roll(self.OLS_buffer, -self.B)
        self.OLS_buffer[self.B+1:2*self.B+1] = x

        # Fast convolution
        y = self.fft_conv(self.OLS_buffer, h)        
        return y[-self.B:]


    def process(self, x, h):       
        # check if the impulse response h has changed
        if np.all(h != self.stored_h):
            self.flagIRchanged = True
            self.stored_h = h
            
        # convolve
        if self.method == 'overlap-save':
            return self.OLS(x, h)
        if self.method == 'overlap-add':
            return self.OLA(x, h)