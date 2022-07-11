# pyFIR
Python implementations of Finite Impulse Response (FIR) filters for real-time convolutions.

The algorithms are mainly *(but not strictly)* the ones described in **WEFERS, Frank. Partitioned convolution algorithms for real-time auralization. Logos Verlag Berlin GmbH, 2015**. found [here](http://publications.rwth-aachen.de/record/466561/files/466561.pdf?subformat=pdfa&version=1).


# Current algorithms 
- Overlap-add (OLA);
- Overlap-save (OLS);
- Uniformily Partitioned Overlap-Save (UPOLS) (generalized version)



*Everything is kinda working at this stage, but keep in mind bugs and mistakes are expected.*

# Requirements
- numpy 

Although to run the example in [testFIR.py](https://github.com/davircarvalho/pyFIR/blob/main/testFIR.py) you will also need:
- librosa
- pyaudio

# TODO
- Filter crossover
