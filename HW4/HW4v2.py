# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:52:10 2020

@author: r07525117
"""

import os
os.chdir(r'D:\ShallowWaterComputation\HW4')
import numpy as np


def eta0(x,y,H=1,L=100):
    return H*np.exp( -18*( x / L )**2 )*np.exp( -18*( y / L )**2 )

def U0(x,y):
    return 0.0

def V0(x,y):
    return 0.0


g = 9.806

xMax = 150
xMin = -150
yMax = 150
yMin = -150

tMax = 20
tMin = 0

deltaX = 5
deltaY = 5
deltaT = 0.05
# CFL = 0.8
# deltaT = CFL*( min(deltaX,deltaY) / np.sqrt(g*10) )

x = np.arange(xMin-2*deltaX, xMax+3*deltaX, deltaX)
y = np.arange(yMin-2*deltaY, yMax+3*deltaY, deltaY)
t = np.arange(0, tMax, deltaT)

[X,Y]=np.meshgrid(x,y)

tlen = len(t)
xlen = len(x)
ylen = len(y)


ETA = np.zeros( ( tlen, ylen, xlen) )
U = np.zeros( ( tlen, ylen, xlen ) )
V = np.zeros( ( tlen, ylen, xlen ) )

ETAs1 = np.zeros( (  ylen, xlen) )
Us1 = np.zeros( (  ylen, xlen ) )
Vs1 = np.zeros( (  ylen, xlen ) )

ETAs2 = np.zeros( (  ylen, xlen) )
Us2 = np.zeros( (  ylen, xlen ) )
Vs2 = np.zeros( (  ylen, xlen ) )

h = np.zeros( ( ylen, xlen) )
h[:] = 10

GN = 2 

##Initialization
ETA[0][GN:-GN, GN:-GN] = eta0( X[GN:-GN, GN:-GN], Y[GN:-GN, GN:-GN] )
U[0][GN:-GN, GN:-GN] = U0( X[GN:-GN, GN:-GN], Y[GN:-GN, GN:-GN] )
V[0][GN:-GN, GN:-GN] = V0( X[GN:-GN, GN:-GN], Y[GN:-GN, GN:-GN] )



n = 0
while( n < (tlen-1) ):
    #BC
    ETA[n][ : , [0,1,-1,-2]  ] = ETA[n][ : , [4,3,-5,-4] ]
    ETA[n][ [0,1,-1,-2] , : ] = ETA[n][ [4,3,-5,-4] , : ]
    
    
    U[n][:,[0,1,-1,-2]] = -U[n][:,[4,3,-5,-4]]  ##L boundary
    U[n][[0,1,-1,-2],:] = U[n][[4,3,-5,-4],:] ##D boundary
    
    V[n][:,[0,1,-1,-2]] = V[n][:,[4,3,-5,-4]]
    V[n][[0,1,-1,-2],:] = -V[n][[4,3,-5,-4],:] 
    
    
    #Cell-first round
    for j in range(2,ylen-2):
        for i in range(2,xlen-2):
            ETAs1[j][i] = ETA[n][j][i] - ( (deltaT) / (12*deltaX) )*\
            ( -U[n][j][i+2]*h[j][i+2] \
            + 8*U[n][j][i+1]*h[j][i+1]\
            -8*U[n][j][i-1]*h[j][i-1] \
            + U[n][j][i-2]*h[j][i-2] )\
            -( (deltaT) / (12*deltaY) )*\
            ( -V[n][j-2][i]*h[j-2][i] \
            +8*V[n][j-1][i]*h[j-1][i]\
            -8*V[n][j+1][i]*h[j+1][i] \
            +V[n][j+2][i]*h[j+2][i] )
    
            Us1[j][i] = U[n][j][i] - ( (deltaT*g) / (12*deltaX) )*\
            (-ETA[n][j][i+2]
            +8*ETA[n][j+1][i+1]\
            -8*ETA[n][j-1][i-1]\
            +ETA[n][j-2][i-2] )
            
            Vs1[j][i] = V[n][j][i] - ( (deltaT*g) / (12*deltaY) )*\
            (-ETA[n][j-2][i]\
            +8*ETA[n][j-1][i]\
            -8*ETA[n][j+1][i]\
            +ETA[n][j+2][i])
    
    #BC
    ETAs1[ : , [0,1,-1,-2]  ] = ETAs1[ : , [4,3,-5,-4] ]
    ETAs1[ [0,1,-1,-2] , : ] = ETAs1[ [4,3,-5,-4] , : ]
    
    Us1[:,[0,1,-1,-2]] = -Us1[:,[4,3,-5,-4]]  ##L boundary
    Us1[[0,1,-1,-2],:] = Us1[[4,3,-5,-4],:] ## D boundary

    Vs1[:,[0,1,-1,-2]] = Vs1[:,[4,3,-5,-4]] ##L boundary
    Vs1[[0,1,-1,-2],:] = -Vs1[[4,3,-5,-4],:] ##D boundary
    
    #Cell-Second round
    for j in range(2,ylen-2):
        for i in range(2,xlen-2):
            ETAs2[j][i] = (3.0/4.0)*ETA[n][j][i]+ (1.0/4.0)*ETAs1[j][i]\
            -( deltaT/(4.0*12*deltaX) )*\
            (-Us1[j][i+2]*h[j][i+2]\
            +8*Us1[j][i+1]*h[j][i+1]\
            -8*Us1[j][i-1]*h[j][i-1]\
            +Us1[j][i-2]*h[j][i-2] )\
            -( deltaT/(4.0*12*deltaY) )*\
            (-Vs1[j-2][i]*h[j-2][i]\
            +8*Vs1[j-1][i]*h[j-1][i]\
            -8*Vs1[j+1][i]*h[j+1][i]\
            +Vs1[j+2][i]*h[j+2][i])
            
            Us2[j][i] = (3.0/4.0)*U[n][j][i] + (1.0/4.0)*Us1[j][i]\
            - ( (deltaT*g) / (4.0*12*deltaX) )*\
            ( -ETAs1[j][i+2]\
             +8*ETAs1[j][i+1]\
             -8*ETAs1[j][i-1]\
             +ETAs1[j][i-2] )
                
            Vs2[j][i]= (3.0/4.0)*V[n][j][i] + (1.0/4.0)*Vs1[j][i]\
            - ( (deltaT*g) / (4.0*12*deltaY) )*\
            ( -ETAs1[j-2][i]\
             +8*ETAs1[j-1][i]\
             -8*ETAs1[j+1][i]\
             +ETAs1[j+2][i] )
    
    #BC
    ETAs2[ : , [0,1,-1,-2]  ] = ETAs2[ : , [4,3,-5,-4] ]
    ETAs2[ [0,1,-1,-2] , : ] = ETAs2[ [4,3,-5,-4] , : ]
    

    Us2[:,[0,1,-1,-2]] = -Us2[:,[4,3,-5,-4]]  ##L boundary
    Us2[[0,1,-1,-2],:] = Us2[[4,3,-5,-4],:] ## D boundary

    Vs2[:,[0,1,-1,-2]] = Vs2[:,[4,3,-5,-4]] ##L boundary
    Vs2[[0,1,-1,-2],:] = -Vs2[[4,3,-5,-4],:] ##D boundary

    #Cell-Third round
    for j in range(2,ylen-2):
        for i in range(2,xlen-2):
            ETA[n+1][j][i] = (1.0/3.0)*ETA[n][j][i] + (2.0/3.0)*ETAs2[j][i]\
            -( (2*deltaT) / (3.0*12*deltaX) )*\
            (-Us2[j][i+2]*h[j][i+2]\
            +8*Us2[j][i+1]*h[j][i+1]\
            -8*Us2[j][i-1]*h[j][i-1]\
            +Us2[j][i-2]*h[j][i-2] )
            -( (2*deltaT) / (3.0*12*deltaY) )*\
            ( -Vs2[j-2][i]*h[j-2][i]\
             +8*Vs2[j-1][i]*h[j-1][i]\
             -8*Vs2[j+1][i]*h[j+1][i]\
             +Vs2[j+2][i]*h[j+2][i] )
                
            U[n+1][j][i] = (1.0/3.0)*U[n][j][i] + (2.0/3.0)*Us2[j][i]\
            - ( (2*deltaT*g)/(3.0*12*deltaX) )*\
            ( -ETAs2[j][i+2]\
            +8*ETAs2[j][i+1]\
            -8*ETAs2[j][i-1]\
            +ETAs2[j][i-2] )
            
            V[n+1][j][i] = (1.0/3.0)*V[n][j][i] + (2.0/3.0)*Vs2[j][i]\
            - ( (2*deltaT*g)/(3.0*12*deltaY) )*\
            ( -ETAs2[j-2][i]\
            +8*ETAs2[j-1][i]\
            -8*ETAs2[j+1][i]\
            +ETAs2[j+2][i] )
    n = n+1
#return x,t,ETA,U,V


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 8))
plt.contourf( X[2:-2,2:-2] ,Y[2:-2,2:-2] , ETA[200][2:-2,2:-2] )
# plt.xlim([-300, 300])
# plt.ylim([-300, 300])
plt.show()


from matplotlib import cm
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X[2:-2, 2:-2], Y[2:-2, 2:-2], ETA[200][2:-2, 2:-2], rstride=1, cstride=1, cmap=cm.viridis)
plt.show()







AAA = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],\
                [13,14,15,16,17,18],\
                [19,20,21,22,23,24],\
                [25,26,27,28,29,30],\
                [31,32,33,34,35,36]])

BB = np.ones((6,6))












