# HCMB

Last update: v1.0

Python code for Wu et al. HCMB: A stable and efficient algorithm for processing the normalization of highly sparse Hi-C contact data

This is an an algorithm, Hi-C Matrix Balancing (HCMB), based on iterative solution of equations, combining with linear search and projection strategy to normalize the Hi-C original interaction data.

Several useful functions:

HCMB

* main code of HCMB algorithm

get_norm_HCMB

* use HCMB to normalize Hi-C contact maps data
  
  input: raw contact matrix n-by-n
  
  output: normlaize matrix, minvalue, maxvalue, running time
    

NormalizeKnightRuizV2

* Python version of Knight Ruiz algorithm for matrix balancing (Knight et al. 2013). This python code is from Rajendra,K. et al. (2017)

  input: raw contact matrix n-by-n
  
  output: normlaize matrix, minvalue, maxvalue
  
and a few functions for plotting and so on.

## Requirements
* Python 3
* numpy
* scipy
* matplotlib
* gcMapExplorer


## Usage
```
from HCMB import get_norm_HCMB, NormalizeKnightRuizV2

# M is nxn matrix type: numpy.ndarray
DM, min, max, rt = get_norm_HCMB(M)
DKR, min, max = NormalizeKnightRuizV2(M)

```


## Reference
Knight,P.A. et al. (2013) A fast algorithm for matrix balancing. IMA Journal of Numerical Analysis, 33, 1029-1047.

Rajendra,K. et al. (2017) Genome contact map explorer: a platform for the comparison,
     interactive visualization and analysis of genome contact maps. Nucleic Acids Research, 45(17):e152.
