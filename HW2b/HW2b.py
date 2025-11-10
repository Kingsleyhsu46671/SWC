# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:33:45 2020

@author: 88693
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
os.chdir('D:\ShallowWaterComputation\HW2b')
from numba import jit

g = 9.806
h = 0.3

def sech(x):
    return 1.0/np.cosh(x)

def eta(x, t=0, H=0.04):
    K = (1.0/h)*np.sqrt( (3*H) / (4*h) )
    C = np.sqrt(g*h)
    return H*np.power( sech( K*( x-C*t ) ) , 2 ) 

def U(x, t=0):
    TEMP = eta(x, t)
    return TEMP *np.sqrt(g*h) /float(h)

def plot(x,y,xlim=None,ylim=None,xn=5,yn=5,xlabel='',ylabel='',filename=None):
    fig, ax1 = plt.subplots()
    ax1.plot( x, y, '-', color = 'black', linewidth= 1)
    ax1.tick_params(which='both',direction='in')
    ax1.minorticks_on()
    ax1.grid()
    if ( xlim!=None ):
        ax1.set_xlim( xlim )
    if ( ylim!=None ):
        ax1.set_ylim( ylim )
    if ( xlabel!='' ):
        ax1.set_xlabel(xlabel, fontsize=20)
    if ( xlabel!='' ):
        ax1.set_ylabel(ylabel, fontsize=20)
    plt.xticks( np.linspace(xlim[0],xlim[1],xn), fontsize=20 )
    plt.yticks( np.linspace(ylim[0],ylim[1],yn), fontsize=20 )
    yfmt = ScalarFormatter()
    yfmt.set_powerlimits((0,0))
    ax1.yaxis.set_major_formatter(yfmt)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig( filename , dpi=300)



@jit
def Numeri(deltaX,End_Time=6.95,Ccfl=1):
    deltaT = float(deltaX)/np.sqrt(g*h)*Ccfl
    
    x = np.arange(-12, 24, deltaX)
    t = np.arange(0, End_Time, deltaT)
    
    tlen = len(t)
    xlen = len(x)
    
    
    ETA = np.empty( ( tlen, xlen+2 ) )
    Vel_U = np.empty( ( tlen, xlen+2 ) )
    
    #####Initialize
    
    ##Cell
    x = np.append(x, x[-1]+deltaX )
    x = np.append(x[0]-deltaX, x )
    
    xlen = len(x)
    
    for i in range(1,xlen-1):
        ETA[0][i] = eta(x[i])
    
    for i in range(1,xlen-1): 
        Vel_U[0][i] = U(x[i])
    

    n = 0
    #######Numerical Solution
    while( n < (tlen-1) ):
        #BC
        ETA[n][0] = ETA[n][2]
        ETA[n][-1] = ETA[n][-3]
        Vel_U[n][0] = 0
        Vel_U[n][-1] = 0
        #Cell
        for i in range(1,xlen-1):
            ETA[n+1][i] = ( ETA[n][i+1] + ETA[n][i-1] )/2.0 - (deltaT*h)/(2*deltaX)*( Vel_U[n][i+1] - Vel_U[n][i-1] ) 
            Vel_U[n+1][i] = ( Vel_U[n][i+1] + Vel_U[n][i-1] ) /2.0 - (deltaT*g)/(2*deltaX)*( ETA[n][i+1] - ETA[n][i-1] )
        n = n+1
    return x, ETA

####Initial Plot
deltaX = 0.00375
x = np.arange(-12, 24, deltaX)
U0 = U(x)
ETA0 = eta(x, t=0, H=0.04)
 
plot(x,ETA0, xlim = [-3,3], ylim = [0,0.04],xlabel='x (m)' ,ylabel='${\eta}$ (m)',filename='ETA0.png')
plot(x,U0, xlim = [-3,3], ylim = [0,0.25],xlabel='x (m)' ,ylabel='${U}$ (m)',filename='U0.png')

####Q1
Cfl = 0.9
x1,ETA1= Numeri(0.00375, Ccfl=Cfl)
x2,ETA2= Numeri(0.0075, Ccfl=Cfl)
x3,ETA3= Numeri(0.015, Ccfl=Cfl)
x4,ETA4= Numeri(0.03, Ccfl=Cfl)
x5,ETA5= Numeri(0.06, Ccfl=Cfl)


#Analytical Solution
x = np.arange(-12, 24, 0.015)
AnalyticalSol = eta(x,t=6.95)

fig, ax1 = plt.subplots()
l1 = ax1.plot( x, AnalyticalSol, '-', color = 'black', linewidth= 1)
l2 = ax1.plot( x1, ETA1[-1], '-', color = 'r', linewidth= 1)
l3 = ax1.plot( x2, ETA2[-1], '--', color = 'r', linewidth= 1)
l4 = ax1.plot( x3, ETA3[-1], '-', color = 'g', linewidth= 1)
l5 = ax1.plot( x4, ETA4[-1], '--', color = 'g', linewidth= 1)
l6 = ax1.plot( x5, ETA5[-1], '-', color = 'b', linewidth= 1)


ax1.set_xlim( [10,14] )
ax1.set_ylim( [0,0.045] )
ax1.tick_params(which='both',direction='in')
ax1.minorticks_on()
ax1.grid()
ax1.set_xlabel( 'x (m)' )
ax1.set_ylabel( '${\eta}$ (m)' )

lns =  l1 + l2 + l3 + l4 + l5 +l6
labels = [  'analytical solution', '${\Delta}X = 0.00375$ (m)' ,'${\Delta}X = 0.0075$ (m)' ,'${\Delta}X = 0.015$ (m)' ,'${\Delta}X = 0.03$ (m)','${\Delta}X = 0.06$ (m)' ]
ax1.legend(lns ,labels , loc = 'lower center' )

fig.tight_layout()
fig.savefig( 'Q1.png', dpi=300)


###L-norm
@jit
def norm(EtaNum, EtaTheory):
    temp = 0
    for i , j in zip(EtaNum, EtaTheory):
        temp = temp + np.power(i-j, 2)
    N = len(EtaNum)
    temp = np.sqrt( float(temp) / float(N) )
    return temp



Cfl = 0.9
x1,ETA1= Numeri(0.015625, Ccfl=Cfl)
x2,ETA2= Numeri(0.03125, Ccfl=Cfl)
x3,ETA3= Numeri(0.0625, Ccfl=Cfl)
x4,ETA4= Numeri(0.125, Ccfl=Cfl)
x5,ETA5= Numeri(0.25, Ccfl=Cfl)

AnalySol1 = eta(x1,t=6.95)
AnalySol2 = eta(x2,t=6.95)
AnalySol3 = eta(x3,t=6.95)
AnalySol4 = eta(x4,t=6.95)
AnalySol5 = eta(x5,t=6.95)


e1 = norm(ETA1[-1],AnalySol1 )
e2 = norm(ETA2[-1],AnalySol2 )
e3 = norm(ETA3[-1],AnalySol3 )
e4 = norm(ETA4[-1],AnalySol4 )
e5 = norm(ETA5[-1],AnalySol5 )

x = np.array([0.015625, 0.03125, 0.0625, 0.125, 0.25])
Error = np.array([e1,e2,e3,e4,e5])

fig, ax1 = plt.subplots()
l1 = ax1.plot( x, Error, '-o', color = 'black', linewidth= 1)

#ax1.set_xlim( [10,14] )
#ax1.set_ylim( [0,0.045] )
ax1.tick_params(which='both',direction='in')
ax1.minorticks_on()
ax1.grid()
ax1.yaxis.get_major_formatter().set_powerlimits((0,2))
ax1.xaxis.get_major_formatter().set_powerlimits((0,1))
ax1.set_xlabel( '${\Delta}x (m)$ ' )
ax1.set_ylabel( '${L^2}-norm (m)$' )


fig.tight_layout()
fig.savefig( 'Q2.png', dpi=300)











