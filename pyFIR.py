'''
pyFIR - 2023
------------------------------------------------------------------------------
Python FIR filters for real-time convolution
------------------------------------------------------------------------------

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
'''

import numpy as np
from numpy.core.numeric import Inf
from numpy.fft import fft, ifft
from copy import deepcopy


class FIRfilter:
    def __init__(self, method="overlap-save", blockSize=512, h=None, partition=None, normalize=True):
        '''
        Performs real-time convolution via FIR filters. This class is a wrapper for the pyFIRfilter()
        which allows for smooth crossover when changing filters at runtime. For most cases this class
        should be prefered.

        Parameters
        ----------
        method : str, optional
            The FIR method to use, available ones are 'overlap-save', 'overlap-add',
            'OLS' (same as overlap-save), 'OLA' (same as overlap-add),
            'UPOLS' (uniformly partitioned overlap-save). The default is "overlap-save".
        blockSize : int, optional
            Block size/buffer size, define the size of the audio chunk being procesed
            at a time unit. The default is 512.
        h : np.array, [samples x channels]
            Impulse response signal to be convolved with the audio input. The default is None.
        partition : int, optional
            Partition size for the UPOLS filter. For general use it is recommended that
            you supply the impulse response instead of the partion size at class initilization,
            by doing this, the optimal partition size will be calculated by default.
        normalize : bool, optional
            Indicate if output should be normalized or not. If True the frame output is going to
            be normalized to be below 1. The default is True.

        Returns
        -------
        FIRfilter.process(x, h) returns the convolution of block x with impulse response h.
        Output has the same size as x and type: np.array

        '''
        self.B = blockSize
        self.currentFIR = pyFIRfilter(method, blockSize, h, partition, normalize)
        self.futureFIR = pyFIRfilter(method, blockSize, h, partition, normalize)
        # self.Xfader = Crossfader(N_cross=B, N_ch=self.N_ch)
        self.current_h = h
        self.flagIRchanged = False
        self.lockXfade = False  # it will only allow for new position after crossfading in done

    def process(self, x, h=None):
        # initialize crossfader now that we know the number of channels in the input
        if not hasattr(self, 'Xfader'):
            if np.ndim(x) == 0:
                Nch = 1
            else:
                Nch = x.shape[-1]
            self.Xfader = Crossfader(N_cross=2*self.B, N_ch=Nch)


        if h is not None and np.any(h != self.current_h):   # check if the impulse response h has changed
            self.flagIRchanged = True

        if self.flagIRchanged and not self.lockXfade:
            self.future_h = h
            self.lockXfade = True
        elif not self.flagIRchanged and not self.lockXfade:
            self.current_h = h
        # elif self.flagIRchanged and self.lockXfade:
        #     pass

        # convolve
        if self.flagIRchanged:  # enable cxfade between past and future filter
            currentOUT = self.currentFIR.process(x, self.current_h)
            futureOUT = self.futureFIR.process(x, self.future_h)
            out, isDone = self.Xfader.process(currentOUT, futureOUT)  # apply crossfading
            self.flagIRchanged = (not isDone)
            if isDone:
                self.currentFIR = deepcopy(self.futureFIR)
                self.current_h = deepcopy(self.future_h)
                self.lockXfade = False
            return out
        else:
            return self.currentFIR.process(x, self.current_h)


class pyFIRfilter:
    '''
    Simple FIR object.
    '''
    def __init__(self, method, blockSize, h, partition, normalize):
        self.method = method.lower()
        self.B = blockSize                # block size (audio input len, which will also be the output size)
        self.stored_h = h        # save IR for comparison next frame (optional input)
        self.partition = partition  # partition size (UPOLS)
        self.normalize = normalize
        self.NFFT = None          # fft/ifft size
        self.flagHchanged = True  # check if the IR changed or it's still the same
        self.Nh = h.shape[0]
        self._generate_ref()

        validMethods = ['overlap-save', 'overlap-add', 'ols', 'ola', 'upols']
        error_msg = f'Unknown FIRfilter method: "{self.method}", \n Supported methods are: {validMethods}'
        assert self.method in validMethods, error_msg

        if ('upols' in self.method) and (partition is None) and (h is not None):
            self.partition, self.NFFT = self._optimize_UPOLS_parameters(self.Nh, self.B)
            # print(f'partition size: {self.partition} \n nfft: {self.NFFT}')

    def _pad_the_end(self, x, new_length):
        if x.shape[0] < new_length:
            output = np.squeeze(np.zeros((new_length, self.N_ch)))
            output[:x.shape[0], ...] = x
        else:
            output = x
        return output

    def _pad_beginning(self, x, padding):
        new_length = x.shape[0] + padding
        output = np.squeeze(np.zeros((new_length, self.N_ch)))
        output[-x.shape[0]:, ...] = x
        return output

    def _generate_ref(self):
        '''
        the worst case scenario given a known IR would be to have a region in the  audio input that's filled with ones,
        this would result in a convolution with output above the 0-1 range.
        This reference below is the safest form of avoiding clipping for any normalized audio input
        '''
        self.ref = np.sqrt(np.max(np.sum(abs(self.stored_h), axis=0)))
        if np.ndim(self.stored_h) > 1:  # just included this logic to update the number of channels
            self.N_ch = self.stored_h.shape[1]
            self.fftAxis = 0  # useful for upols only
        else:
            self.N_ch = 1
            self.fftAxis = 0  # useful for upols only

    def _normalize_output(self, data):
        return data / self.ref

    def _optimize_UPOLS_parameters(self, N, B):
        '''brute-force the optimal parameters for UPOLS
            N: IR length
            B: buffer size
        '''
        def cost(B, N, L, K):
            # theoretical time estimates for each operation, pag. 211
            return 1 / B * (1.68 * K * np.log2(K) + 3.49 * K * np.log2(K) + 6 * ((K + 1) / 2) + ((N / L) - 1) * 8 * ((K + 1) / 2))

        c_opt = Inf
        rang = np.array([2**k for k in range(0, int(np.log2(N)))]).astype('int')
        for L in rang:
            d_max = B - np.gcd(L, B)
            K_min = int(B + L + d_max - 1)
            K_max = int(2**np.ceil(np.log2(K_min)))
            k_sort = np.sort([K_min, K_max])
            if k_sort[0] == k_sort[1]:
                k_sort[1] = k_sort[1] + 1

            K_range = np.arange(k_sort[0], k_sort[1] + 1)
            for K in K_range:
                c = cost(B, N, L, K)
                if c < c_opt:
                    c_opt = c
                    L_opt = L
                    K_opt = K
        return L_opt, K_opt

    def _fft_conv(self, x):
        X = fft(x, self.NFFT, axis=0)
        if self.flagHchanged:  # store the IR fft
            self.H = fft(self.stored_h, self.NFFT, axis=0)
            self.flagHchanged = False
        return ifft(X * self.H, axis=0).real

    def _OLA(self, x):
        if self.NFFT is None:
            self.NFFT = self.B + self.stored_h.shape[0] - 1
            self.left_overs = np.squeeze(np.zeros((self.NFFT, self.N_ch)))
            self.len_y_left = self.NFFT - self.B

        # Fast convolution
        y = self._fft_conv(x)

        # Overlap-Add the partial convolution result
        out = y[:self.B, ...] + self.left_overs[:self.B, ...]

        self.left_overs = np.roll(self.left_overs, -self.B, axis=0)  # flush the buffer
        self.left_overs[-self.B:, ...] = 0
        self.left_overs[:self.len_y_left, ...] = self.left_overs[:self.len_y_left, ...] + y[self.B:, ...]
        if self.normalize:
            return self._normalize_output(out)
        else:
            return out

    def _OLS(self, x):
        if self.NFFT is None:
            self.NFFT = self.B + self.Nh - 1
            # Input buffer
            self.OLS_input_buffer = np.squeeze(np.zeros((self.NFFT, self.N_ch)))

        # Sliding window of the input
        self.OLS_input_buffer = np.roll(self.OLS_input_buffer, shift=-self.B, axis=0)  # previous contents are shifted B samples to the left
        self.OLS_input_buffer[-self.B:, ...] = x  # next length-B input block is stored rightmost

        # Fast convolution
        out = self._fft_conv(self.OLS_input_buffer)

        if self.normalize:
            return self._normalize_output(out[-self.B:, ...])
        else:
            return out[-self.B:, ...]

    def _UPOLS(self, x, h):
        if self.flagHchanged:  # only run on on initial call
            Nh = self.Nh
            if self.partition is None:
                print('Running UPOLS parameter optimization, this may take a few minutes to run deppending on the length of the impulse respose, to avoid this optimization prcedure, simply declare a "partition" value at the class initialization')
                self.partition, self.NFFT = self._optimize_UPOLS_parameters(Nh, self.B)
                L_partit = self.partition
            else:
                L_partit = self.partition
                dmax = self.B - np.gcd(L_partit, self.B)
                self.NFFT = self.B + L_partit + dmax

            self.P = int(np.ceil(Nh / L_partit))  # number of partitions done
            self.input_buffer = np.squeeze(np.zeros((self.NFFT, self.N_ch)))  # Initialize input buffer
            # Initialize filter and FDL
            self.nm = np.zeros((self.P,))  # tells us which FDL positions should be used
            self.H = np.zeros((self.P, self.NFFT, self.N_ch), dtype='complex_')  # partitioned filters (freq domain)
            if self.N_ch == 1:
                self.H = np.squeeze(self.H, axis=-1)

            # (1) split original filter into P length-L sub filters
            for m, ii in enumerate(range(0, Nh, L_partit)):
                try:
                    h_partit = h[ii:(ii + L_partit), ...]
                except Exception:
                    h_partit = self._pad_the_end(h[ii:, ...], L_partit)

                # (2) incorporate "remainder delays"
                dm = np.mod(m * L_partit, self.B)
                h_pad = self._pad_beginning(h_partit, dm)
                self.H[m, ...] = fft(h_pad, n=self.NFFT, axis=0)
                self.nm[m] = np.floor(m * L_partit / self.B)  # FDL active slots
            self.nm = self.nm.astype(int)
            self.FDL = np.zeros((max(self.nm) + 1, self.NFFT, self.N_ch), dtype='complex_')  # delay line
            if self.N_ch == 1:
                self.FDL = np.squeeze(self.FDL, axis=-1)

            if np.ndim(self.FDL) == 1:
                self.FDL = np.expand_dims(self.FDL, axis=0)  # case self.nm==0

            self.flagHchanged = False

        # (3) Sliding window of the input
        self.input_buffer = np.roll(self.input_buffer, shift=-self.B, axis=0)  # previous contents are shifted B samples to the left
        self.input_buffer[-self.B:, ...] = x  # next length-B input block is stored rightmost
        # (4) Stream
        self.FDL = np.roll(self.FDL, shift=1, axis=0)  # shift the FDL to create space for current input
        self.FDL[0, ...] = fft(self.input_buffer, axis=self.fftAxis)  # add current buffer to the first FDL slot
        # convo
        # note: the sum is done in the frequency domain (yep!)
        out = ifft(np.sum(self.FDL[self.nm, ...] * self.H, axis=0), axis=self.fftAxis).real

        if self.normalize:
            return self._normalize_output(out[-self.B:, ...])
        else:
            return out[-self.B:]

    def process(self, x, h=None):
        if (h is not None) and (np.any(h != self.stored_h)):   # check if the impulse response h has changed
            self.flagHchanged = True
            self.stored_h = h
        else:
            h = self.stored_h

        if self.method == 'overlap-save' or self.method == 'ols':
            return self._OLS(x)
        elif self.method == 'overlap-add' or self.method == 'ola':
            return self._OLA(x)
        elif self.method == 'upols':
            return self._UPOLS(x, h)


# %% Crossfading
class Crossfader:
    def __init__(self, N_cross=2048, N_ch=None):
        '''
        N_cross: durantion of the crossover in samples
        N_ch: Number of channels to crossover
        '''
        self.isDone = False  # if any xfade was performed or not
        self.reset_faders = True
        self.N_cross = N_cross
        self.N_ch = N_ch
        self._reset_fader_funcs()

    def process(self, A, B):
        '''
        Parameters
        ----------
        A : numpy nd array
            Audio processed with impulse response at previous state.
        B : numpy nd array
           Audio at processed with impulse response future/target state.

        Returns
        -------
        C  : numpy nd array
            Audio output at current crossfade step.

        '''
        if self.reset_faders:
            self._reset_fader_funcs()

        frame_sz = A.shape[0]
        # (1) - Case the length of the xfade is longer than the frame
        if self.xfade_up.shape[0] > frame_sz:
            C = (A * self.xfade_down[:frame_sz, ...]) + (B * self.xfade_up[:frame_sz, ...])
            # Delete the fader part that was already used
            self.xfade_up = np.delete(self.xfade_up, range(frame_sz), axis=0)
            self.xfade_down = np.delete(self.xfade_down, range(frame_sz), axis=0)
            self.isDone = False

        # (2) - Case the length of the xfade is smaller than the frame
        elif 0 < self.xfade_up.shape[0] < frame_sz:
            frame_sz = int(self.xfade_up.shape[0])
            C = (A[:frame_sz, ...] * self.xfade_down) + (B[:frame_sz, ...] * self.xfade_up)
            C = np.concatenate((C, B[frame_sz:, ...]), axis=0)

            # Delete the fader part that was already used
            self.xfade_up = np.delete(self.xfade_up, range(frame_sz), axis=0)
            self.xfade_down = np.delete(self.xfade_down, range(frame_sz), axis=0)
            self.isDone = True
            self.reset_faders = True

        # (3) - Case the crossfading is already done but didn't catch it
        else:
            self.reset_faders = True
            return B, True
        return C, self.isDone


    def _reset_fader_funcs(self):
        # Ensure the fade functions have the same dimentions as the input
        self.xfade_up = np.expand_dims(np.linspace(0, 1, self.N_cross), axis=-1)
        self.xfade_down = np.expand_dims(np.linspace(1, 0, self.N_cross), axis=-1)
        self.xfade_up = np.tile(self.xfade_up, self.N_ch)
        self.xfade_down = np.tile(self.xfade_down, self.N_ch)
        self.reset_faders = False
