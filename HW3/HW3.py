# -* coding: utf-8 -*-
"""
Created on Mon Mar 30 11:37:18 2020

@author: 88693
"""


import os
os.chdir('D:\ShallowWaterComputation\HW3')
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

g = 9.81

def sech(x):
    return 1.0/np.cosh(x)

def eta(x, h=0.1, t=0, H=0.04):
    K = (1.0/h)*np.sqrt( (3*H) / (4*h) )
    C = np.sqrt(g*h)
    return H*np.power( sech( K*( x-C*t ) ) , 2 ) 

def U(x, h=0.1, t=0):
    TEMP = eta(x,h,t)
    return TEMP *np.sqrt(g*h) / float(h)

def h(x , a = 8):
    return 0.3-0.2*( 1 + np.tanh( a*(x-12) ) )/2.0
# def h(x):
#     return 0.3

Ccfl = 0.9

@jit
def Numeric(deltaX,End_Time):
    th = 0.3
    deltaT = float(deltaX)/np.sqrt(g*th)*Ccfl
    
    x = np.arange(-12, 24, deltaX)
    t = np.arange(0, End_Time, deltaT)

    tlen = len(t)
    xlen = len(x)+4
    
    ETA = np.zeros( ( tlen, xlen ) )
    Vel_U = np.zeros( ( tlen, xlen ) )
    
    ETAs1 = np.zeros( ( tlen, xlen ) )
    Vel_Us1 = np.zeros( ( tlen, xlen ) )
    
    ETAs2 = np.zeros( ( tlen, xlen ) )
    Vel_Us2 = np.zeros( ( tlen, xlen ) ) 
    
    ###get ghost cell and BCs
    x = np.append(x, [ x[-1]+deltaX, x[-1]+2*deltaX ] )
    x = np.append( [ x[0]-2*deltaX, x[0]-deltaX ], x )
    
    ##Initialization
    for i in range(2,xlen-2):
        ETA[0][i] = eta( x[i], h(x[i]) )
    
    for i in range(2,xlen-2): 
        Vel_U[0][i] = U( x[i], h(x[i]) )
    
    n = 0
    while( n < (tlen-1) ):
        #BC
        ETA[n][0] = ETA[n][4]
        ETA[n][1] = ETA[n][3]
        ETA[n][-1] = ETA[n][-5]
        ETA[n][-2] = ETA[n][-4]
        Vel_U[n][0] = 0
        Vel_U[n][1] = 0
        #Vel_U[n][2] = 0 
        Vel_U[n][-1] = 0
        Vel_U[n][-2] = 0
        #Vel_U[n][-3] = 0
        #Cell
        for i in range(2,xlen-2):
            ETAs1[n][i] = ETA[n][i] - ( (deltaT)/(12*deltaX) )*( -Vel_U[n][i+2]*h(x[i+2]) + \
            8*Vel_U[n][i+1]*h(x[i+1]) - 8*Vel_U[n][i-1]*h(x[i-1]) + Vel_U[n][i-2]*h(x[i-2]) ) 
            Vel_Us1[n][i] = Vel_U[n][i] - ( (deltaT*g)/(12*deltaX) ) * \
            ( -ETA[n][i+2] + 8*ETA[n][i+1] - 8*ETA[n][i-1] + ETA[n][i-2] )
        ETAs1[n][0] = ETAs1[n][4]
        ETAs1[n][1] = ETAs1[n][3]
        ETAs1[n][-1] = ETAs1[n][-5]
        ETAs1[n][-2] = ETAs1[n][-4]
        Vel_Us1[n][0] = 0
        Vel_Us1[n][1] = 0
        #Vel_Us1[n][2] = 0 
        Vel_Us1[n][-1] = 0
        Vel_Us1[n][-2] = 0
        #Vel_Us1[n][-3] = 0
        for i in range(2,xlen-2):
            ETAs2[n][i] = (3.0/4.0)*ETA[n][i] + (1.0/4.0)*ETAs1[n][i] - ( deltaT/(4.0*12*deltaX) ) \
            * ( -Vel_Us1[n][i+2]*h(x[i+2]) + 8*Vel_Us1[n][i+1]*h(x[i+1]) \
            -8*Vel_Us1[n][i-1]*h(x[i-1]) + Vel_Us1[n][i-2]*h(x[i-2]) )
            Vel_Us2[n][i] = (3.0/4.0)*Vel_U[n][i] + (1.0/4.0)*Vel_Us1[n][i] - ( (deltaT*g) / (4*12*deltaX) ) \
            *( -ETAs1[n][i+2] + 8*ETAs1[n][i+1] - 8*ETAs1[n][i-1] + ETAs1[n][i-2] )
        ETAs2[n][0] = ETAs2[n][4]
        ETAs2[n][1] = ETAs2[n][3]
        ETAs2[n][-1] = ETAs2[n][-5]
        ETAs2[n][-2] = ETAs2[n][-4]
        Vel_Us2[n][0] = 0
        Vel_Us2[n][1] = 0
        #Vel_Us2[n][2] = 0 
        Vel_Us2[n][-1] = 0
        Vel_Us2[n][-2] = 0
        #Vel_Us2[n][-3] = 0
        for i in range(2,xlen-2):
            ETA[n+1][i] = (1.0/3.0)*ETA[n][i] + (2.0/3.0)*ETAs2[n][i] - ( ( 2.0*deltaT )/( 3.0*12*deltaX ) )* \
            ( - Vel_Us2[n][i+2]*h(x[i+2]) + 8*Vel_Us2[n][i+1]*h(x[i+1]) - 8*Vel_Us2[n][i-1]*h(x[i-1]) \
            + Vel_Us2[n][i-2]*h(x[i-2]) )
            Vel_U[n+1][i] = (1.0/3.0)*Vel_U[n][i] + (2.0/3.0)*Vel_Us2[n][i] - ( (2.0*deltaT*g)/(3*12*deltaX) ) \
            * ( -ETAs2[n][i+2] + 8*ETAs2[n][i+1] -8*ETAs2[n][i-1] + ETAs2[n][i-2] )
        n = n+1
    return x,t,ETA,Vel_U
    

x1,t1,Et1,Vu1 = Numeric(0.03,6.95) 
x2,t2,Et2,Vu2 = Numeric(0.06,6.95) 
x3,t3,Et3,Vu3 = Numeric(0.12,6.95)
x4,t4,Et4,Vu4 = Numeric(0.24,6.95)  

#Analytical Solution
x = np.arange(-12, 24, 0.03)
AnalyticalSol = eta(x,h=0.3,t=6.95)


fig, ax1 = plt.subplots()

l1 = ax1.plot(x ,AnalyticalSol, '-', color = 'black', linewidth= 1 )
l2 = ax1.plot( x1, Et1[-1], '-', color = 'r', linewidth= 1)
l3 = ax1.plot( x2, Et2[-1], '--', color = 'r', linewidth= 1)
l4 = ax1.plot( x3, Et3[-1], '-', color = 'b', linewidth= 1)       
l5 = ax1.plot( x4, Et4[-1], '--', color = 'b', linewidth= 1)            
ax1.set_xlim([9,15])    
ax1.grid(linestyle = '--')   
ax1.tick_params(which='both',direction='in')
ax1.set_xlabel( 'x (m)' )
ax1.set_ylabel( '${\eta}$ (m)' )

lns =  l1 + l2 + l3 + l4+ l5
labels = [  'Analytical solution', '${\Delta}X = 0.03$ (m)', '${\Delta}X = 0.06$ (m)' \
          ,'${\Delta}X = 0.12$ (m)' , '${\Delta}X = 0.24$ (m)']
ax1.legend(lns ,labels , loc = 'upper left',prop={'size':8} )

fig.tight_layout()
fig.savefig( 'Q1' , dpi=300)        
        
        
###L-norm
@jit
def norm(EtaNum, EtaTheory):
    temp = 0
    for i , j in zip(EtaNum, EtaTheory):
        temp = temp + np.power(i-j, 2)
    N = len(EtaNum)
    temp = np.sqrt( float(temp) / float(N) )
    return temp

 #Analytical Solution  
x1,t1,Et1,Vu1 = Numeric(0.03,6.95) 
x2,t2,Et2,Vu2 = Numeric(0.06,6.95) 
x3,t3,Et3,Vu3 = Numeric(0.12,6.95)
x4,t4,Et4,Vu4 = Numeric(0.24,6.95)  

AnalySol1 = eta(x1,h=0.3,t=t1[-1])   
AnalySol2 = eta(x2,h=0.3,t=t2[-1])   
AnalySol3 = eta(x3,h=0.3,t=t3[-1])      
AnalySol4 = eta(x4,h=0.3,t=t4[-1])   
    
e1 = norm(Et1[-1],AnalySol1 )  
e2 = norm(Et2[-1],AnalySol2 )
e3 = norm(Et3[-1],AnalySol3 )
e4 = norm(Et4[-1],AnalySol4 )


x = np.array([ 0.000027,0.000216, 0.001728, 0.013824])
Error = np.array([e1,e2,e3,e4])

fig, ax1 = plt.subplots()
l1 = ax1.plot( x, Error, '-o', color = 'black', linewidth= 1)

#ax1.set_xlim( [10,14] )
#ax1.set_ylim( [0,0.045] )
ax1.tick_params(which='both',direction='in')
ax1.minorticks_on()
ax1.grid()
ax1.yaxis.get_major_formatter().set_powerlimits((0,2))
ax1.xaxis.get_major_formatter().set_powerlimits((0,1))
ax1.set_xlabel( '${\Delta}x^3 (m)$ ' )
ax1.set_ylabel( '${L^2}-norm (m)$' )


fig.tight_layout()
fig.savefig( 'Q2.png', dpi=300)


####Q3
x1,t1,Et1,Vu1 = Numeric(0.03,13.9) 
fig, ax1 = plt.subplots()
y = h(x1)

l1 = ax1.plot( x1, Et1[-1]+0.3, '-', color = 'r', linewidth= 1)
l2 = ax1.plot( x1, y, '-', color = 'black', linewidth= 1)

ax1.set_xlim([5,15]) 


HI = 0.04
HT = max(Et1[-1]) #0.050963
HR = max(Et1[-1][0:600]) ##0.00986167

HT_HI = 1.274075
HR_HI = 0.24654




    
