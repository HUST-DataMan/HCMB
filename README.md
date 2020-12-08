# HCMB
This is an an algorithm, Hi-C Matrix Balancing (HCMB), based on iterative solution of equations, combining with linear search and projection strategy to normalize the Hi-C original interaction data.

Several useful functions:

HCMB
   1. main code of HCMB algorithm

get_norm_HCMB
   1. use HCMB to normalize Hi-C contact maps data
    input: raw matrix
    output: normlaize matrix, minvalue, maxvalue, running time
    

NormalizeKnightRuizV2
    1. python version of Knight Ruiz algorithm for matrix balancing (Knight et al. 2013)
     This python code is from Rajendra,K. et al. (2017)
and a few functions for plotting and so on.


# Reference
Knight,P.A. et al. (2013) A fast algorithm for matrix balancing. IMA Journal of Numerical Analysis, 33, 1029-1047.

Rajendra,K. et al. (2017) Genome contact map explorer: a platform for the comparison,
     interactive visualization and analysis of genome contact maps. Nucleic Acids Research, 45(17):e152.
