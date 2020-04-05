# -*- coding: utf-8 -*-
"""
Created on Wed May 9 12:43:49 2018

Time of simulations

Addition
1) Vertex counter for each iteration
2) Display domain state like MFM images
3) 64x64 array : 3E6 iterations : 23min16sec
@author: vineeth
"""

import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
from copy import copy, deepcopy
from numpy import unravel_index
from IPython import get_ipython

from Function_ASI_Energy import ASI_Lattice
from Function_ASI_Energy import sort_vertex_type
from Function_ASI_Energy import sort_vertex_btype
from Function_ASI_Energy import asi_energy
from Function_Display_ASI import DomainImage
#get_ipython().run_line_magic('matplotlib', 'qt')
starttime = datetime.datetime.now()

#==== Lattice initialization=============================================
#ASI array size
Xcell, Ycell = Size_Vertex_matrix+1, Size_Vertex_matrix+1 # actual number of squares cells forming the ASI array >> Vertex matrix size= (Xcell-1)x(Ycell-1)
M, N = 2*Xcell+1, 2*Ycell+1  ## Matrix size representing the ASI array for calculations and representations 
its = 300 # Total number of iterations
#S, Sxy, Sx, Sy = ASI_Lattice(M,N,0) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]

#============Control switches==================================================
dipolar_switch = 1
exchange_switch = 0
Exchange_Bias = 0
graphs = 1

#================ External parameters =========================================
#T = 0.10  # System temperature
#kb = 1 # 1.38064852e-23 # Boltman constant
#b = 3  # inverse temperature 1/kT typical=1
Ha = 0.01 # Applied magnetic field Hc ~ 1.30
phi_H = 0 # Applied magnetic field direction
H = (Ha*math.cos(math.radians(phi_H)), Ha*math.sin(math.radians(phi_H))) # Applied magnetic field
C = 1  #1e-7 # Dipolar interaction constant mu/4pi

#============ASI intrinsic properties = Energy barrier distribution, Exchange bias==============================
Gaussian_distribution = np.random.normal(.70, 0.001, size=(M*N,))
switching_Barrier = np.reshape(Gaussian_distribution, (M, N))
Ebxy = np.multiply(switching_Barrier,abs(Sx+Sy))
Exy = np.zeros((M,N))
Ed, Ez, Ex, Eeb = np.zeros((M,N)), np.zeros((M,N)), np.zeros((M,N)), np.zeros((M,N))

if (Exchange_Bias == 0):
    h_eb = 1.0 # Exchange bias magnitude
    dh_eb = 0.010 # Standard deviation
    phi_Heb = 180 # Angle in degrees
    dphi_Heb = 2 # Standard deviation in direction of exchange bias
    Exchange_Barrier = np.reshape(np.random.normal(h_eb,dh_eb, size=(M*N,)), (M, N))
    h_eb_arr = np.multiply(Exchange_Barrier,abs(Sx+Sy))
    Exchange_angle_arr = np.reshape(np.random.normal(math.radians(phi_Heb), math.radians(dphi_Heb), size=(M*N,)), (M, N))
    Hxy_eb_arr = np.stack([h_eb_arr*np.cos(Exchange_angle_arr),h_eb_arr*np.sin(Exchange_angle_arr)], axis=0)
else:
    Hxy_eb_arr =  np.stack([0*Sx, 0*Sy], axis=0)

#================ Initialize variables of the simulation =======================================================
ITERATION = [] # np.zeros(its) # iterations
avg_Mx = []#np.zeros(its) # X-sublattice Order parameter
avg_My = []#np.zeros(its) # Y-sublattice Order parameter
Vtype1 = []
Vtype2 = []
Vtype3 = []
Vtype4 = []
Vtype10, Vtype11, Vtype20, Vtype21, Vtype22, Vtype23, Vtype30, Vtype31, Vtype32, Vtype33, Vtype34, Vtype35, Vtype36, Vtype37, Vtype40, Vtype41  = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]



j=0 #iteration number
ITERATION.append(0)

#======= Counting vertices ====================================================
Vertex = np.argwhere(S == 0)#[0]
Cl,Cr,Ct,Cd = 0,0,0,0
V = np.zeros((M,N))
Vb = np.zeros((M,N))
V1 = np.zeros((M,N))
E1=0
for ze in Vertex:
    xi = ze[1]
    yi = ze[0]
    if (1<xi<M-2 and 1<yi<N-2):
        Cl = Sxy[0,yi,xi-1]
        Cr = Sxy[0,yi,xi+1]
        Cd = Sxy[1,yi+1,xi]
        Ct = Sxy[1,yi-1,xi]        
        if     ((abs(Cl) == 1 and abs(Cr) == 1))\
            and ((abs(Cd) == 1 and abs(Ct) == 1)):

            if (Cl+Cr+Ct+Cd == 2) or (Cl+Cr+Ct+Cd == -2):
                Vtype = 3
                if (Cl==1 and Ct==1 and Cr==1 and Cd==-1):
                    Vbtype = 30
                if (Cl==1 and Ct==-1 and Cr==1 and Cd==1):
                    Vbtype = 31
                if (Cl==1 and Ct==1 and Cr==-1 and Cd==1):
                    Vbtype = 32                    
                if (Cl==1 and Ct==-1 and Cr==-1 and Cd==-1):
                    Vbtype = 33                    
                if (Cl==-1 and Ct==1 and Cr==1 and Cd==1):
                    Vbtype = 34                    
                if (Cl==-1 and Ct==-1 and Cr==1 and Cd==-1):
                    Vbtype = 35                    
                if (Cl==-1 and Ct==1 and Cr==-1 and Cd==-1):
                    Vbtype = 36                    
                if (Cl==-1 and Ct==-1 and Cr==-1 and Cd==1):
                    Vbtype = 37                    
                                        
            if (Cl+Cr+Ct+Cd == 4 or Cl+Cr+Ct+Cd == -4) or (Cl+Ct == 0 and Cr+Cd == 0):
                Vtype = 2
                if ((Cl==1 and Ct==1) and (Cr==1 and Cd==1)):
                    Vbtype = 20
                if ((Cl==1 and Ct==-1) and (Cr==1 and Cd==-1)):
                    Vbtype = 21
                if ((Cl==-1 and Ct==1) and (Cr==-1 and Cd==1)):
                    Vbtype = 22
                if ((Cl==-1 and Ct==-1) and (Cr==-1 and Cd==-1)):
                    Vbtype = 23
                
            if (Cr+Ct == 2 and Cl+Cd == -2):
                Vtype = 4
                Vbtype = 41                
            if (Cr+Ct == -2 and Cl+Cd == 2):
                Vtype = 4
                Vbtype = 40
                
            if (Cl+Ct == 2 and Cr+Cd == -2):
                Vtype = 1
                Vbtype = 10
            if (Cl+Ct == -2 and Cr+Cd == 2):
                Vtype = 1
                Vbtype = 11
                
    else:
        Vtype = 0
        Vbtype = 0        
    V[yi,xi] = Vtype
    V1[yi,xi] = Vtype
    Vb[yi,xi] = Vbtype

A = V.flatten()
B = Vb.flatten()

Vtype10.append(np.count_nonzero(B == 10))
Vtype11.append(np.count_nonzero(B == 11))
Vtype20.append(np.count_nonzero(B == 20))
Vtype21.append(np.count_nonzero(B == 21))
Vtype22.append(np.count_nonzero(B == 22))
Vtype23.append(np.count_nonzero(B == 23))
Vtype30.append(np.count_nonzero(B == 30))
Vtype31.append(np.count_nonzero(B == 31))
Vtype32.append(np.count_nonzero(B == 32))
Vtype33.append(np.count_nonzero(B == 33))
Vtype34.append(np.count_nonzero(B == 34))
Vtype35.append(np.count_nonzero(B == 35))
Vtype36.append(np.count_nonzero(B == 36))
Vtype37.append(np.count_nonzero(B == 37))
Vtype40.append(np.count_nonzero(B == 40))
Vtype41.append(np.count_nonzero(B == 41))

Vtype1.append(np.count_nonzero(A == 1))
Vtype2.append(np.count_nonzero(A == 2))
Vtype3.append(np.count_nonzero(A == 3))
Vtype4.append(np.count_nonzero(A == 4))
NSx = (N-1)*(M+1)/4 # Total number of spins in Y-sublattice
NSy = (M-1)*(N+1)/4 # Total number of spins in X-sublattice
NS  = NSx*NSy # Total number of spins


#====== Calculate energy ======================================================
E_tot=0
Ed_tot=0
for i in range (0,N):
    for j in range (0,M):        
        Exy[j,i], Ed[j,i], Ez[j,i], Ex[j,i], Eeb[j,i]  = asi_energy(i,j,Sxy,M,N,H,C,dipolar_switch,exchange_switch, Exchange_Bias, Hxy_eb_arr)
        E_tot = E_tot + Exy[j,i]
        Ed_tot = Ed_tot + Ed[j,i] 
                
E_avg = E_tot/(M*N)
Ed_avg = Ed_tot/(M*N)
print('Average initial energy = ', E_avg)
print('Average initial dipolar energy = ', Ed_avg)

####====================== Display initial state===============================
if (graphs==1):    
    plt.interactive(True) 
    plt.ion()
    Intial = plt.figure(1)
    ASI_Intial = plt.figure(1)
    plt.imshow(S, interpolation='nearest')
    plt.colorbar()
    Intial.show()
    rx,ry = [],[]
    rx,ry = np.meshgrid(np.arange(0, M, 1), np.arange(0, N, 1))
    plt.quiver(rx, ry, Sx, Sy,pivot='mid')
    ASI_Intial.show()
    ASI_Intial.savefig('ASI_Initialstate.png')

    InitialDomainState = DomainImage(Sxy[0,:,:], Sxy[1,:,:])
    InitialDomainState.save('InitialDomainState.png')
    InitialDomainState.show()

###=========Display energy map=================================================
    plt.interactive(True) 
    plt.ion()
    energy_plot = plt.figure(2)
    plt.clf()
    plt.imshow(Exy, interpolation='nearest')
    plt.colorbar()
    #plt.pause(0.000001)
    energy_plot.show()
    energy_plot.savefig('Initial_energy_plot.png')

#======= Initial average magnetization in X- and Y- sublattices================
rws, clm = np.nonzero(Sxy[0,:,:])
Sx = Sxy[0,:,:]
nonZeroSx = Sx[rws, clm]
avg_Mx.append(np.mean(nonZeroSx)/NSx)
rws, clm = np.nonzero(Sxy[1,:,:])
Sy = Sxy[1,:,:]
nonZeroSy = Sy[rws, clm]
my = np.mean(nonZeroSy)
avg_My.append(np.mean(nonZeroSy)/NSy)

#=================== Main calculations ========================================
for i in range (1,its):
    if (i%2 == 0):
        xi = unravel_index(Exy.argmax(), Exy.shape)[1]
        yi = unravel_index(Exy.argmax(), Exy.shape)[0]    
#    else:        
#        xi = random.randint(0,N-1)  # choose randomly from the MxN lattice the position of a spin
#        yi = random.randint(0,M-1)  # N colums x M rows        

    R = (yi,xi)  #Position vector of spin "M"
    mi = Sxy[:,yi,xi] 
# mi= (mx,my); mx = Sxy[0,yi,xi] and my = Sx[1,yi,xi]
    if np.dot(mi,mi)!=0 or np.dot(mi,mi)>0:
        Exy[yi,xi], Ed[yi,xi], Ez[yi,xi], Ex[yi,xi], Eeb[yi,xi]   = asi_energy(xi,yi,Sxy,M,N,H,C,dipolar_switch,exchange_switch,Exchange_Bias, Hxy_eb_arr)            
#        (Exy[yi,xi], Ed[yi,xi], Ez[yi,xi], Ex[yi,xi], Eeb[yi,xi])  = asi_energy(i,j,Sxy,M,N,H,C,dipolar_switch,exchange_switch, Exchange_Bias, Hxy_eb_arr)
        #        p = random.uniform(0, 1)            
        if (Exy[yi,xi] >= 1*Ebxy[yi,xi]):# and np.exp((E)*b) > p): # dE = 2Ee Energy difference
            Sxy[:,yi,xi] = -1*Sxy[:,yi,xi]
#        else: # Thermal fluctuations
#            p = random.uniform(0, 1)
#            if(np.exp((E+E0)*b) > p): 
#                Sxy[:,yi,xi] = -1*Sxy[:,yi,xi]
        if (Sxy[0,yi,xi] != 0): # x compoment
            if (0<xi<N-1):
                vert_right = xi+1
                vert_left  = xi-1
                V[yi,xi+1] = sort_vertex_type(yi,vert_right,Sxy,M,N)                
                V[yi,xi-1] = sort_vertex_type(yi,vert_left,Sxy,M,N)
                Vb[yi,xi+1] = sort_vertex_btype(yi,vert_right,Sxy,M,N)                
                Vb[yi,xi-1] = sort_vertex_btype(yi,vert_left,Sxy,M,N)
        if (Sxy[1,yi,xi] != 0):
            if (0<yi<M-1):
                vert_top = yi-1
                vert_bot = yi+1
                V[yi+1,xi] = sort_vertex_type(vert_bot,xi,Sxy,M,N)
                V[yi-1,xi] = sort_vertex_type(vert_top,xi,Sxy,M,N)
                Vb[yi+1,xi] = sort_vertex_btype(vert_bot,xi,Sxy,M,N)
                Vb[yi-1,xi] = sort_vertex_btype(vert_top,xi,Sxy,M,N)
                 
        j = j+1        
        ITERATION.append(j)
        rws, clm = np.nonzero(Sxy[0,:,:])
        Sx = Sxy[0,:,:]
        nonZeroSx = Sx[rws, clm]
        avg_Mx.append(np.sum(nonZeroSx)/NSx)
        rws, clm = np.nonzero(Sxy[1,:,:])
        Sy = Sxy[1,:,:]
        nonZeroSy = Sy[rws, clm]
        my = np.mean(nonZeroSy)
        avg_My.append(np.sum(nonZeroSy)/NSy)
        Vtype1.append(np.count_nonzero(V.flatten() == 1))
        Vtype2.append(np.count_nonzero(V.flatten() == 2))
        Vtype3.append(np.count_nonzero(V.flatten() == 3))
        Vtype4.append(np.count_nonzero(V.flatten() == 4))
        Exy[yi,xi], Ed[yi,xi], Ez[yi,xi], Ex[yi,xi], Eeb[yi,xi]  = asi_energy(xi,yi,Sxy,M,N,H,C,dipolar_switch,exchange_switch, Exchange_Bias, Hxy_eb_arr)

        Vtype10.append(np.count_nonzero(Vb.flatten() == 10))
        Vtype11.append(np.count_nonzero(Vb.flatten() == 11))
        Vtype20.append(np.count_nonzero(Vb.flatten() == 20))
        Vtype21.append(np.count_nonzero(Vb.flatten() == 21))
        Vtype22.append(np.count_nonzero(Vb.flatten() == 22))
        Vtype23.append(np.count_nonzero(Vb.flatten() == 23))
        Vtype30.append(np.count_nonzero(Vb.flatten() == 30))
        Vtype31.append(np.count_nonzero(Vb.flatten() == 31))
        Vtype32.append(np.count_nonzero(Vb.flatten() == 32))
        Vtype33.append(np.count_nonzero(Vb.flatten() == 33))
        Vtype34.append(np.count_nonzero(Vb.flatten() == 34))
        Vtype35.append(np.count_nonzero(Vb.flatten() == 35))
        Vtype36.append(np.count_nonzero(Vb.flatten() == 36))
        Vtype37.append(np.count_nonzero(Vb.flatten() == 37))
        Vtype40.append(np.count_nonzero(Vb.flatten() == 40))
        Vtype41.append(np.count_nonzero(Vb.flatten() == 41))        
#=======End of caculation =====================================================


#========== Outputs of calculations============================================= 
S = Sxy[0,:,:] + Sxy[1,:,:]
np.savetxt('Sx.txt', Sxy[0,:,:])
np.savetxt('Sy.txt', Sxy[1,:,:])

E_tot=0
Ed_tot=0
for i in range (0,N):
    for j in range (0,M):        
        Exy[j,i], Ed[j,i], Ez[j,i], Ex[j,i], Eeb[j,i]  = asi_energy(i,j,Sxy,M,N,H,C,dipolar_switch,exchange_switch, Exchange_Bias, Hxy_eb_arr)
        E_tot = E_tot + Exy[j,i]
        Ed_tot = Ed_tot + Ed[j,i] 
        
        
E_avg = E_tot/(M*N)
Ed_avg = Ed_tot/(M*N)
print('Average final energy = ', E_avg)
print('Average final dipolar energy = ', Ed_avg)

if (graphs == 1):    
    plt.interactive(True)
    plt.ion()
    ASI_Final = plt.figure(3)
    plt.imshow(S, interpolation='nearest')
    plt.colorbar()
    rx,ry = np.meshgrid(np.arange(0, M, 1), np.arange(0, N, 1))
    plt.quiver(rx, ry,Sxy[0,:,:], Sxy[1,:,:], pivot='mid')#, line=dict(width=1))#  scale=.01)#, arrow_scale=.4, name='quiver', line=dict(width=1))
    ASI_Final.show()
    ASI_Final.savefig('QFinalstate.png')

    plt.interactive(True)
    plt.ion()    
    energy_plot = plt.figure(4)
    plt.clf()
    plt.imshow(Exy, interpolation='nearest')
    plt.colorbar()
    energy_plot.show()
    energy_plot.savefig('Final_energy_plot.png')

#=========== Order parameter vs time ==========================================
    f = plt.figure(5)
    plt.axis([0,its,-1,1])
    plt.clf()
    plt.semilogx(ITERATION, avg_Mx, ITERATION, avg_My)
    f.show()
    plt.savefig('Average_M.jpg')
    f.canvas.draw()
    
    FinalDomainState = DomainImage(Sxy[0,:,:], Sxy[1,:,:])
    FinalDomainState.save('FinalDomainState.png')
    FinalDomainState.show()
    
finaltime = datetime.datetime.now()
time_taken = finaltime - starttime
print('Time taken = ', time_taken)


