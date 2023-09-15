# FIRconv

<p align="left">
  <a href="https://github.com/davircarvalho/FIRconv/releases/" target="_blank">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/davircarvalho/FIRconv?include_prereleases&style=flat-square">
  </a>

  <a href="https://github.com/davircarvalho/FIRconv/commits/master" target="_blank">
    <img src="https://img.shields.io/github/last-commit/davircarvalho/FIRconv?style=flat-square" alt="GitHub last commit">
  </a>

  <a href="https://github.com/davircarvalho/FIRconv/issues" target="_blank">
    <img src="https://img.shields.io/github/issues/davircarvalho/FIRconv?style=flat-square&color=red" alt="GitHub issues">
  </a>

  <a href="https://github.com/davircarvalho/FIRconv/blob/master/LICENSE" target="_blank">
    <img alt="LICENSE" src="https://img.shields.io/github/license/davircarvalho/FIRconv?style=flat-square&color=yellow">
  <a/>

</p>
<hr>


Python implementation of real-time convolution algorithms for Finite Impulse Response (FIR) filters.

The algorithms are mainly *(but not strictly)* the ones described in **WEFERS, Frank. Partitioned convolution algorithms for real-time auralization. Logos Verlag Berlin GmbH, 2015**. found [here](http://publications.rwth-aachen.de/record/466561/files/466561.pdf?subformat=pdfa&version=1).


# Current algorithms
- Overlap-add (OLA);
- Overlap-save (OLS);
- Uniformily Partitioned Overlap-Save (UPOLS) (generalized/optimized version)

# Installation
Use pip to install FIRconv:
```
$ pip install FIRconv
```

# Getting started
Bellow there's a pseudo-code showing how to setup a basic use of FIRconv for real time convolutions.

```python
from FIRconv import FIRfilter

# Initialize FIR filter
bufferSize = 2**10
method = 'upols'
FIRfilter(method, bufferSize, h=IR)

while 1:
  output = FIRfilter.process(audio)
  play(output)
```


- For more in-depth examples have a look at [testFIR.py](https://github.com/davircarvalho/FIRconv/blob/main/testFIR.py) or [Algorithms_validation.py](https://github.com/davircarvalho/FIRconv/blob/main/Algorithms_validation.py)
___________________________________________________________________
Collaborations are more than welcome!
