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
from numpy.core.fromnumeric import partition
from numpy.fft import fft, ifft


class FIRfilter():
    '''
        method:  'overlap-save'(default) or 'overlap-add'
        B: defines the block length
        h: optional call, initializes the impulse response to be convolved
    '''
    
    def __init__(self, method="overlap-save", B=512, h=None, partition=512):    
        self.method = method
        self.B = B                # block size (audio input len, which will also be the output size)
        self.NFFT = None          # fft/ifft size
        self.flagIRchanged = True # check if the IR changed or it's still the same
        self.stored_h = h         # save IR for comparison next frame (optional input)
        self.left_overs = np.zeros((B,)) # remaining samples from last ifft (OLA)
        self.partition = partition # partition size (UPOLS)       
        
        
    def len(x):
        return max(np.shape(x))
                

    def pad_the_end(self, x, new_length):
        output = np.zeros((new_length,))
        output[:x.shape[0]] = x
        return output

    
    def pad_beginning(self, x, padding):
        new_length = max(x.shape) + padding
        output = np.zeros((new_length,))
        output[-x.shape[0]:] = x
        return output
    

    def next_power_of_2(self, n):
        return 1 << (int(np.log2(n - 1)) + 1)


    def fft_conv(self,x, h):
        # Calculate x fft
        X = fft(x, self.NFFT)
        # Calculate h fft
        if self.flagIRchanged: # store the IR fft
            self.H = fft(h, self.NFFT)
            self.flagIRchanged = False
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
        self.left_overs = self.pad_the_end(self.left_overs,pad) + y[self.B+1:] 
        if len(self.left_overs) < self.B: # fix size for next loop (if needed)
            self.left_overs = self.pad_the_end(self.left_overs, self.B)      
        return out


    def OLS(self, x, h):
        if self.NFFT is None:
            Nx = max(x.shape)
            Nh = max(h.shape)
            self.NFFT = self.next_power_of_2(Nx + Nh - 1) 
            # Input buffer
            self.OLS_buffer = np.zeros(shape=(self.NFFT,))        
        
        # Sliding window of the input
        self.OLS_buffer = np.roll(self.OLS_buffer, -self.B) #previous contents are shifted B samples to the left
        self.OLS_buffer[self.B:2*self.B] = x #next length-B input block is stored rightmost

        # Fast convolution
        y = self.fft_conv(self.OLS_buffer, h)     
        return y[-self.B:]

    
    def UPOLS(self,x,h):
        '''(generalized) Uniformly Partitioned Overlap-Save
            - generalized means that you can pick the partition size
        '''
        if self.NFFT is None: # only run on on initial call 
            L_partit = self.partition # sub filter length (partition length)
            dmax = self.B - np.gcd(L_partit, self.B)           
            self.NFFT = self.B + L_partit + dmax
            Nh = max(h.shape)
            self.P = int(np.ceil(Nh/L_partit)) # number of partitions done
            # Input buffer
            self.input_buffer = np.zeros(shape=(self.NFFT,))                
            # variables for delay line
            self.nm = np.zeros((self.P,)) # tells us which FDL should be used
            self.FDL = np.zeros((self.P, self.NFFT), dtype='complex_')  # delay line 
            self.H = np.zeros((self.P, self.NFFT),dtype='complex_') # partitioned filters (freq domain)
            
            # (1) split original filter into P length-L sub filters  
            for m, ii in enumerate(range(0, Nh, L_partit)):
                try:
                    h_partit = h[ii : ii+L_partit]
                except:
                    print('except')
                    h_partit = self.pad_the_end(h[ii:], L_partit)
                    
                # (2) incorporate "remainder delays"         
                dm = np.mod(m*L_partit, self.B)         
                h_pad = self.pad_the_end(self.pad_beginning(h_partit, dm), self.NFFT)  
                self.H[m,:] = fft(h_pad, n=self.NFFT)    
                self.nm[m] = np.floor(m*L_partit/self.B)  # FDL active slots        
            self.nm = self.nm.astype(int)

        # (3) Sliding window of the input
        self.input_buffer = np.roll(self.input_buffer, shift=-self.B) #previous contents are shifted B samples to the left
        self.input_buffer[-self.B:] = x #next length-B input block is stored rightmost     
        # (4) Stream 
        self.FDL = np.roll(self.FDL, shift=1, axis=0) # shift the FDL to create space for current input  
        self.FDL[0,:] = fft(self.input_buffer) # add current buffer to the first FDL slot
        # convo
        # note: the sum is done in the frequency domain (yep!)
        out = ifft(np.sum(np.multiply(self.FDL[self.nm,:], self.H), axis=0),n=self.NFFT).real
        return out[-self.B:]
                  
                  
                  
# %% Main     
    def process(self, x, h):       
        # check if the impulse response h has changed
        if np.all(h != self.stored_h):
            self.flagIRchanged = True
            self.stored_h = h
        else:
            h = self.stored_h             
            
        # convolve
        if self.method=='overlap-save' or self.method=='OLS':
            return self.OLS(x, h)
        elif self.method=='overlap-add' or self.method=='OLA':
            return self.OLA(x, h)
        elif self.method=='UPOLS':
            return self.UPOLS(x, h)