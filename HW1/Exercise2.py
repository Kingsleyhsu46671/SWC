# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:40:41 2020

@author: 88693
"""

import numpy as np


x = np.arange(-1, 3, 0.01)
t = np.arange(0, 2, 0.01)


g = 9.81

h = 0.1

def eta(x):
    return 0.02*np.exp( -8*np.pow(x,2) )


ETA = np.empty( ( 0 , 0 ) )

tlen = len(t)
xlen = len(x)

deltaT = 0.01
deltaX = 0.01

for i in range(tlen):
        (eta(i+1)+eta(i-1)) / 2.0  - 




