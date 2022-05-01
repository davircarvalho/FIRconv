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
from numpy.core.numeric import Inf
from numpy.fft import fft, ifft
from time import time 

class FIRfilter():
    '''
        method:  'overlap-save'(default) or 'overlap-add'
        B: defines the block length
        h: optional call, initializes the impulse response to be convolved
    '''

    def __init__(self, method="overlap-save", B=512, h=None, partition=None):    
        self.method = method
        self.B = B                # block size (audio input len, which will also be the output size)
        self.NFFT = None          # fft/ifft size
        self.flagIRchanged = True  # check if the IR changed or it's still the same
        self.stored_h = h         # save IR for comparison next frame (optional input)
        self.left_overs = np.zeros((B,))  # remaining samples from last ifft (OLA)
        self.partition = partition  # partition size (UPOLS)

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
        '''Overlap-add convolution'''
        if self.NFFT is None:
            Nx = max(x.shape)
            Nh = max(h.shape)
            self.NFFT = self.next_power_of_2(Nx + Nh - 1)  

        # Fast convolution
        y = self.fft_conv(x, h)

        # Overlap-Add the partial convolution result        
        out = y[:self.B] + self.left_overs[:self.B]

        self.left_overs = np.delete(self.left_overs, [range(0, self.B)], 0)    
        pad = len(y[self.B+1:])
        self.left_overs = self.pad_the_end(self.left_overs, pad) + y[self.B+1:] 
        if len(self.left_overs) < self.B:  # fix size for next loop (if needed)
            self.left_overs = self.pad_the_end(self.left_overs, self.B)      
        return out

    def OLS(self, x, h):
        '''Overlap-save convolution'''
        if self.NFFT is None:
            Nx = max(x.shape)
            Nh = max(h.shape)
            self.NFFT = self.next_power_of_2(Nx + Nh - 1) 
            # Input buffer
            self.OLS_buffer = np.zeros(shape=(self.NFFT,))        

        # Sliding window of the input
        self.OLS_buffer = np.roll(self.OLS_buffer, -self.B)  # previous contents are shifted B samples to the left
        self.OLS_buffer[self.B:2*self.B] = x  # next length-B input block is stored rightmost

        # Fast convolution
        y = self.fft_conv(self.OLS_buffer, h)     
        return y[-self.B:]

    def optimize_UPOLS_parameters(self, N, B):
        '''brute-force the optimal parameters for UPOLS
            N: IR length
            B: buffer size
        '''
        def cost(B, N, L, K):
            '''theoretical time estimates for each operation, pag. 211'''
            # def T_fft(n):
            #     return 1.68 *n*np.log2(n)
            # def T_ifft(n):
            #     return 3.49 *n*np.log2(n)
            # def T_cmul(n):
            #     return 6*n
            # def T_cmac(n):
            #     return 8*n   
            # return 1/B*(T_fft(K) + T_ifft(K) + T_cmul((K+1)/2) + ((N/L)-1)*T_cmac((K+1)/2))
            return 1/B*(1.68*K*np.log2(K) + 3.49*K*np.log2(K) + 6*((K+1)/2) + ((N/L)-1)*8*((K+1)/2))
        
        c_opt = Inf
        rang = np.array([2**k for k in range(0, int(np.log2(N)))]).astype('int')
        for L in rang:
            d_max = B - np.gcd(L, B)
            K_min = int(B + L + d_max - 1)
            K_max = int(2**np.ceil(np.log2(K_min)))
            k_sort = np.sort([K_min, K_max])
            if k_sort[0] == k_sort[1]:
                k_sort[1] = k_sort[1]+1
            
            for K in range(k_sort[0], k_sort[1]):
                c = cost(B, N, L, K)
                if c < c_opt:
                    c_opt = c
                    L_opt = L
                    K_opt = K
        return L_opt, K_opt


    def UPOLS(self, x, h):
        '''(generalized) Uniformly Partitioned Overlap-Save
            - generalized means that you can pick the partition size
        '''
        if self.NFFT is None: # only run on on initial call 
            Nh = max(h.shape)
            if self.partition is None:
                print('Running UPOLS parameter optimization, this may take a few minutes to run deppending on the length of the impulse respose, to avoid this optimization prcedure, simply declare a "partition" value at the class initialization')
                self.partition, self.NFFT = self.optimize_UPOLS_parameters(Nh, self.B)
                L_partit = self.partition
            else:
                L_partit = self.partition
                dmax = self.B - np.gcd(L_partit, self.B)  
                self.NFFT = self.B + L_partit + dmax

            self.P = int(np.ceil(Nh/L_partit))  # number of partitions done
            self.input_buffer = np.zeros(shape=(self.NFFT,))  # Initialize input buffer
            # Initialize filter and FDL
            self.nm = np.zeros((self.P,))  # tells us which FDL positions should be used
            self.H = np.zeros((self.P, self.NFFT), dtype='complex_')  # partitioned filters (freq domain)

            # (1) split original filter into P length-L sub filters
            for m, ii in enumerate(range(0, Nh, L_partit)):
                try:
                    h_partit = h[ii:ii+L_partit]
                except:
                    h_partit = self.pad_the_end(h[ii:], L_partit)

                # (2) incorporate "remainder delays"
                dm = np.mod(m*L_partit, self.B)
                h_pad = self.pad_beginning(h_partit, dm)
                self.H[m, :] = fft(h_pad, n=self.NFFT)
                self.nm[m] = np.floor(m*L_partit/self.B)  # FDL active slots
            self.nm = self.nm.astype(int)
            self.FDL = np.zeros((max(self.nm)+1, self.NFFT), dtype='complex_')  # delay line

        # (3) Sliding window of the input
        self.input_buffer = np.roll(self.input_buffer, shift=-self.B) #previous contents are shifted B samples to the left
        self.input_buffer[-self.B:] = x  # next length-B input block is stored rightmost     
        # (4) Stream 
        self.FDL = np.roll(self.FDL, shift=1, axis=0)  # shift the FDL to create space for current input
        self.FDL[0, :] = fft(self.input_buffer)  # add current buffer to the first FDL slot
        # convo
        # note: the sum is done in the frequency domain (yep!)
        out = ifft(np.sum(np.multiply(self.FDL[self.nm, :], self.H), axis=0)).real
        return out[-self.B:]
    
    
    
    # def NUPOLS(self,x,h):
    #     '''Non Uniformly Partitioned Overlap-Save
    #     '''
    #     def sub_UPOLS(self,xi,hi,Pi,L_partit):
    #         '''
    #         xi: High grain discretization of the original input x
    #         Si: Filter segment
    #         Pi: Number of filter partitions in the current segment
    #         L_partit: Lenght of the current sub-filter partition
    #         Bi: length of the xi input
    #         '''
    #         Nh = max(hi.shape)
    #         Bi = Nh
    #         NFFT = 2*Nh            
    #         input_buffer = np.zeros(shape=(NFFT,))   # Initialize input buffer              
    #         # Initialize filter and FDL
    #         nm = np.zeros((Pi,)) # tells us which FDL positions should be used 
    #         H = np.zeros((Pi, NFFT),dtype='complex_') # partitioned filters (freq domain)
            
    #         # (1) split original filter into P length-L sub filters  
    #         for m, ii in enumerate(range(0, Nh, L_partit)):
    #             try:
    #                 h_partit = hi[ii:ii+L_partit]
    #             except:
    #                 h_partit = self.pad_the_end(hi[ii:], L_partit)
                    
    #             # (2) incorporate "remainder delays"         
    #             dm = np.mod(m*L_partit, Bi)         
    #             h_pad = self.pad_the_end(self.pad_beginning(h_partit, dm), NFFT)  
    #             H[m,:] = fft(h_pad, n=NFFT)    
    #             nm[m] = np.floor(m*L_partit/Bi)  # FDL active slots        
    #         nm = nm.astype(int)
    #         FDL = np.zeros((max(nm)+1, NFFT), dtype='complex_')  # delay line 
            
    #         # (4) sub-Stream 
    #         # Sliding window of the input
    #         input_buffer = np.roll(input_buffer, shift= -Bi) #previous contents are shifted B samples to the left
    #         input_buffer[-Bi:] = xi #next length-B input block is stored rightmost     
    #         FDL = np.roll(FDL, shift=1, axis=0) # shift the FDL to create space for current input  
    #         FDL[0,:] = fft(input_buffer) # add current buffer to the first FDL slot
    #         # convo
    #         # note: the sum is done in the frequency domain (yep!)
    #         out = ifft(np.sum(np.multiply(FDL[nm,:], H), axis=0)).real
    #         return out[-Bi:]                 
        # (1) Stablish filter segment sizes
        
            
                  
# %% Main ###############################################################################    
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