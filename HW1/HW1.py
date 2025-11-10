# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:01:37 2020

@author: 88693
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os
os.chdir('D:\ShallowWaterComputation\HW1')

###Calculate Fourier series Sum
def FouriSum(n, t=0, A0 = 0.5, L0=1, Xdelta = 0.005, h0=0.05 ):
    X = np.arange(-1,1,Xdelta)
    fx = 0
    g = 9.806
    for n in range(1,n+1):
        Kn = ( (2*n-1)*2*np.pi / L0 )
        Wn =  np.sqrt( g * Kn * np.tanh(Kn*h0) )
        fx = fx + ( 1/(2*n-1) )*np.sin(X*Kn-Wn*t)
    fx = fx * ( ( 4*A0 ) / np.pi )
    return X, fx

x,y1=FouriSum(3)
x,y2=FouriSum(10)
x,y3=FouriSum(50)

#spline fitting plot to make the cure more smooth
xnew = np.linspace(x.min(), x.max(), 3000)  
power_smooth1 = make_interp_spline(x, y1)(xnew)
power_smooth2 = make_interp_spline(x, y2)(xnew)
power_smooth3 = make_interp_spline(x, y3)(xnew)

#plot figure
fig, ax1 = plt.subplots()
linwid = 1
l1 = ax1.plot( xnew , power_smooth1 , '-' , color = 'r' , linewidth= linwid)
l2 = ax1.plot( xnew , power_smooth2 , '-' , color = 'g' , linewidth= linwid)
l3 = ax1.plot( xnew , power_smooth3 , '-' , color = 'b' , linewidth= linwid)

#setting grid
ax1.tick_params(which='both',direction='in')
ax1.minorticks_on()
ax1.grid()

#Setting label
ax1.set_xlabel( 'x (m)' )
ax1.set_ylabel( 'f(x) (m)' )

#setting plot legend 
lns = l1 + l2 + l3 
labels = [  '3 terms' , '10 terms' ,'50 terms'   ]
ax1.legend(lns ,labels , loc = 1 )

#output plot
fig.tight_layout()
fig.savefig( 'Q1.png', dpi=300)

#####Q2

x,y1=FouriSum(3 ,t = 0 )
x,y2=FouriSum(3 ,t = 0.25 )
x,y3=FouriSum(3 ,t = 0.5 )

#spline fitting plot to make the cure more smooth
xnew = np.linspace(x.min(), x.max(), 3000)  
power_smooth1 = make_interp_spline(x, y1)(xnew)
power_smooth2 = make_interp_spline(x, y2)(xnew)
power_smooth3 = make_interp_spline(x, y3)(xnew)

#plot figure
fig, ax1 = plt.subplots()
linwid = 1
l1 = ax1.plot( xnew , power_smooth1 , '-' , color = 'r' , linewidth= linwid)
l2 = ax1.plot( xnew , power_smooth2 , '-' , color = 'g' , linewidth= linwid)
l3 = ax1.plot( xnew , power_smooth3 , '-' , color = 'b' , linewidth= linwid)

#setting grid
ax1.tick_params(which='both',direction='in')
ax1.minorticks_on()
ax1.grid()

ax1.set_xlabel( 'x (m)' )
ax1.set_ylabel( '$\eta$ (m)' )


lns = l1 + l2 + l3 
labels = [  't=0s' , 't=0.25s' ,'t=0.5s'   ]
ax1.legend(lns ,labels , loc = 'lower right' )

#output plot
fig.tight_layout()
fig.savefig( 'Q2.png', dpi=300)














