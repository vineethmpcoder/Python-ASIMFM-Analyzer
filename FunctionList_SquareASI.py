# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:14:47 2017

@author: vineeth
Important note: This is collection of the different functions used in the main program
Function list and their outputs
1. frange(start, stop, step) ==>> return(arr)
2. ASI_Lattice(M,N,State) ==>> return (S,Sxy,Sx,Sy)
3. sort_vertex_type(yii,xii,Sxy) ==>> return (Vtype)
4. sort_vertex_btype(yii,xii,Sxy) ==>> return (Vbtype,Vtype)
5. Vertex_count_Sxy(Sxy) ==>>  return (Vertex_type_list,Vb,V)
6. total_dipolarenergy(IndexArray_non_zeros,size_0Sx,Sxy) ==>> return (Ed_tot/2)
7. totalenergy_mi(Sxy,xi,yi,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag) ==>> return(Ei_tot,C*Edi, mag*Ezi, mag*Eebi)
8. totalenergy(Sxy,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag) ==>> return(E_tot)
9. EnergyBarrier_arr(Sxy,mag,hb,dhb) ==>> return (Ebxy)
10. Exchange_bias_arr(Sxy,h_eb,dh_eb, phi_Heb, dphi_Heb) ==>> return (Hxy_eb_arr)
11. MCsimulation(MC_steps,MC_cycles,M,N,IS,hb,dhb,H,Exchange_Bias,h_eb,dh_eb,phi_Heb, dphi_Heb,T,C,mag) ==>>     return(Eav,C_T,Mx,My,VertexType_count,VertexType_std_count)
12. MC_onecyle_simulation(MC_steps,M,N,IS,hb,dhb,H,Exchange_Bias,h_eb,dh_eb,phi_Heb, dphi_Heb,T,C,mag) ==>>      return(Eav,Mx,My,VertexType_count,ITERATION,E_it,Sxy)
13. createheader(Xcell,Ycell,MC_steps,MC_cycles,IS,mag,lattice_spacing,C,Ha,phi_H,Temperatures,T,Heb_list,dh_eb,phi_Heb,dphi_Heb,hb,dhb) ==>> return(head)
14. DomainImage(Sxy) ==>> return(domain_image)
15. Vertex_map(Sxy) ==>> return(cv_img)
16. MatrixImport(Fileloc) ==>> return(moment)
Time taken for 
its      arrray

10x10
10,000    42 sec 
40,000    3 min 30 sec 

20x20
10,000    3 min 9 sec 
30,000    8 min 47 sec 
40,000    11 min 53 sec 
50,000   15 min 45 sec

30x30
1e5    1:08:04.213872 not saturated
1e5    1:55:34.844829 not saturated
1e5    1:39:08.495617 saturated

40x40
1e6    1 day, 17:07:14.340795 saturated/close

"""

import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
from copy import copy, deepcopy
from IPython import get_ipython
import os
import cv2
#from PIL import Image
#get_ipython().run_line_magic('matplotlib', 'qt')


def frange(start, stop, step): # Input start, stop and step value and return list of values
    decimal_places = str(step)[::-1].find('.')
    arr = []
    i = start
    while i < stop:
         arr.append(i)
         i += step
         i = round(i,decimal_places)
    return(arr)  
#####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

####=====Generate different ASI configurations=================================================
def ASI_Lattice(M,N,State):
    a = np.ones(N, dtype=int)
    a[::2]=0
    a = a.reshape(N)
    b = np.zeros(N, dtype=int)
    b[::2]=1
    L = deepcopy(a)
    S = np.zeros((M,N))
    for d in range(0,M-1):
        if d % 2 == 0: 
            L = np.row_stack((L,b))
        else:
            L = np.row_stack((L,a))
            
#######======= Different intial states
#####Intialize the random ASI spin array
    if (State == 0):
        S = np.random.rand(M,N)  
        S[S>0.5] = 1
        S[S<0.5] = -1

#Intialize spin array to +M
    elif (State == 1):
        S = np.ones((M,N))      #Intialize spin array to +M

# True antiferromagnetic state    
    else: 
        for i in range (0,N):
            if(i%2==0 and (i//2)%2==0):
                for j in range (0,M):                   
                    if(j%2==0):
                        S[i,j] = 0                
                    elif(j%2 ==1):
                        if((j//2)%2 == 0 or j==1):# and (i//2)%2==0):
                            S[i,j] = 1
                        else:
                            S[i,j] = -1
                            
            elif(i%2==0 and (i//2)%2==1):
                for j in range (0,M):                   
                    if(j%2==0):
                        S[j,i] = 0                
                    elif(j%2 ==1):
                        if((j//2)%2 == 0 or j==1):# and (i//2)%2==0):
                            S[i,j] = -1
                        else:
                            S[i,j] = 1
                
            elif(i%2==1 and (i//2)%2==0):
                for j in range (0,M):
                    if(j%2==1):
                        S[j,i] = 0                   
                    elif(j%2==0):
                        if((j//2)%2 == 0 or j==0):# and (i//2)%2==0):
                            S[i,j] = 1
                        else:
                            S[i,j] = -1            
            else:
                for j in range (0,M):
                    if(j%2==1):
                        S[j,i] = 0                   
                    elif(j%2==0):
                        if((j//2)%2 == 0 or j==0):# and (i//2)%2==0):
                            S[i,j] = -1
                        else:
                            S[i,j] = 1

    S = S*L
    Sy = deepcopy(S)
    Sy[::2] = 0
    Sx = deepcopy(S)
    Sx[1::2] = 0
    Sxy = np.stack([Sx, Sy], axis=0)
    return (S,Sxy,Sx,Sy)
####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&





####=======Sort type of vertex at a given index==================================================
def sort_vertex_type(yii,xii,Sxy): ## *&* Index of the vertex and not of spin
    M,N = np.shape(Sxy)[1], np.shape(Sxy)[2]
    Vtype = 0
    if (1<xii<M-2 and 1<yii<N-2):
        Cl = Sxy[0,yii,xii-1]
        Cr = Sxy[0,yii,xii+1]
        Cd = Sxy[1,yii+1,xii]
        Ct = Sxy[1,yii-1,xii]        
        if     ((abs(Cl) == 1 and abs(Cr) == 1))\
            and ((abs(Cd) == 1 and abs(Ct) == 1)):        
            if (Cl+Cr+Ct+Cd == 2 or Cl+Cr+Ct+Cd == -2):
                Vtype = 3        
            if (Cl+Cr+Ct+Cd == 4 or Cl+Cr+Ct+Cd == -4) or (Cl+Ct == 0 and Cr+Cd == 0):
                Vtype = 2           
            if ((Cr+Ct == 2 and Cl+Cd == -2) or (Cr+Ct == -2 and Cl+Cd == 2)):
                Vtype = 4                
            if (Cl+Ct == 2 and Cr+Cd == -2) or (Cl+Ct == -2 and Cr+Cd == 2):
                Vtype = 1         
    else:
        Vtype = 0        
    return (Vtype)

def sort_vertex_btype(yii,xii,Sxy): ## *&* Index of the vertex and not of spin
    M,N = np.shape(Sxy)[1], np.shape(Sxy)[2]
    Vbtype = 0
    Vtype = 0
    if (1<xii<M-2 and 1<yii<N-2):
        Cl = Sxy[0,yii,xii-1]
        Cr = Sxy[0,yii,xii+1]
        Cd = Sxy[1,yii+1,xii]
        Ct = Sxy[1,yii-1,xii]        
        if     ((abs(Cl) == 1 and abs(Cr) == 1))\
            and ((abs(Cd) == 1 and abs(Ct) == 1)):                
            if (Cl+Cr+Ct+Cd == 2) or (Cl+Cr+Ct+Cd == -2):
                Vtype = 3
                if (Cl == 1 and Cr == 1 and Ct == 1 and Cd == -1):
                    Vbtype = 30
                if (Cl == 1 and Cr == 1 and Ct == -1 and Cd == 1):
                    Vbtype = 31
                if (Cl == -1 and Cr == 1 and Ct == 1 and Cd == 1):
                    Vbtype = 32                    
                if (Cl == 1 and Cr == -1 and Ct == 1 and Cd == 1):
                    Vbtype = 33                    
                if (Cl == -1 and Cr == 1 and Ct == -1 and Cd == -1):
                    Vbtype = 34                    
                if (Cl == 1 and Cr == -1 and Ct == -1 and Cd == -1):
                    Vbtype = 35                    
                if (Cl == -1 and Cr == -1 and Ct == 1 and Cd == -1):
                    Vbtype = 36                    
                if (Cl == -1 and Cr == -1 and Ct == -1 and Cd == 1):
                    Vbtype = 37                    
                                        
            if (Cl+Cr+Ct+Cd == 4 or Cl+Cr+Ct+Cd == -4) or (Cl+Ct == 0 and Cr+Cd == 0):
                Vtype = 2
                if (Cl==1 and Cr==1 and Ct==1 and Cd==1):
                    Vbtype = 20
                if (Cl==1 and Cr==1 and Ct==-1 and Cd==-1):
                    Vbtype = 21
                if (Cl==-1 and Cr==-1 and Ct==-1 and Cd==-1):
                    Vbtype = 22
                if (Cl==-1 and Cr==-1 and Ct==1 and Cd==1):
                    Vbtype = 23
                
            if (Cr+Ct == 2 and Cl+Cd == -2):
                Vtype = 4
                Vbtype = 40                
            if (Cr+Ct == -2 and Cl+Cd == 2):
                Vtype = 4
                Vbtype = 41
                
            if (Cl+Ct == 2 and Cr+Cd == -2):
                Vtype = 1
                Vbtype = 10
            if (Cl+Ct == -2 and Cr+Cd == 2):
                Vtype = 1
                Vbtype = 11
    else:
        Vtype = 0
        Vbtype = 0
    return (Vbtype,Vtype)
####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def Sxy_to_Vxy(Sxy):    
    Vxy = np.zeros((Sxy.shape[1],Sxy.shape[2]))
    for xi in range (0,Sxy.shape[1]-1):
        for yi in range (0,Sxy.shape[2]-1):
            if (Sxy[0,yi,xi] != 0): # x compoment
                if (0<xi<Sxy.shape[1]-1):
                    vert_right = xi+1
                    vert_left  = xi-1
                    Vxy[yi,xi+1] = sort_vertex_btype(yi,vert_right,Sxy)                
                    Vxy[yi,xi-1] = sort_vertex_btype(yi,vert_left,Sxy)
    
            if (Sxy[1,yi,xi] != 0):
                if (0<yi<Sxy.shape[2]-1):
                    vert_top = yi-1
                    vert_bot = yi+1
                    Vxy[yi+1,xi] = sort_vertex_btype(vert_bot,xi,Sxy)
                    Vxy[yi-1,xi] = sort_vertex_btype(vert_top,xi,Sxy)
    return (Vxy)

#####====Identify all vertices in Sxy====
def Vertex_count_Sxy(Sxy): # Count the vertices in Sxy in percentage value
    M = np.shape(Sxy)[1]
    N = np.shape(Sxy)[2]
    V1,V2,V3,V4 = 0,0,0,0
    V10,V11,V20,V21,V22,V23,V30,V31,V32,V33,V34,V35,V36,V37,V40,V41 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    Vertex_type_list = []
    #======= Counting vertices ====================================================
    Vertex = np.argwhere(Sxy[0,:,:]+Sxy[1,:,:] == 0)#[0]
    V = np.zeros((M,N))
    Vb = np.zeros((M,N))
    for ze in Vertex:
        xi = ze[1]
        yi = ze[0]
        if (0<xi<N-1):
            V[yi,xi] = sort_vertex_type(yi,xi,Sxy)
            Vb[yi,xi],V[yi,xi] = sort_vertex_btype(yi,xi,Sxy)
        if (0<yi<M-1):
            V[yi,xi] = sort_vertex_type(yi,xi,Sxy)
            Vb[yi,xi],V[yi,xi] = sort_vertex_btype(yi,xi,Sxy)
    Vtotal = ((M-1)/2-1)*((N-1)/2-1)
    V1 = np.count_nonzero(V.flatten() == 1)*100/Vtotal
    V2 = np.count_nonzero(V.flatten() == 2)*100/Vtotal
    V3 = np.count_nonzero(V.flatten() == 3)*100/Vtotal
    V4 = np.count_nonzero(V.flatten() == 4)*100/Vtotal
    
    V10 = np.count_nonzero(Vb.flatten() == 10)*100/Vtotal
    V11 = np.count_nonzero(Vb.flatten() == 11)*100/Vtotal
    V20 = np.count_nonzero(Vb.flatten() == 20)*100/Vtotal
    V21 = np.count_nonzero(Vb.flatten() == 21)*100/Vtotal
    V22 = np.count_nonzero(Vb.flatten() == 22)*100/Vtotal
    V23 = np.count_nonzero(Vb.flatten() == 23)*100/Vtotal
    V30 = np.count_nonzero(Vb.flatten() == 30)*100/Vtotal
    V31 = np.count_nonzero(Vb.flatten() == 31)*100/Vtotal
    V32 = np.count_nonzero(Vb.flatten() == 32)*100/Vtotal
    V33 = np.count_nonzero(Vb.flatten() == 33)*100/Vtotal
    V34 = np.count_nonzero(Vb.flatten() == 34)*100/Vtotal
    V35 = np.count_nonzero(Vb.flatten() == 35)*100/Vtotal
    V36 = np.count_nonzero(Vb.flatten() == 36)*100/Vtotal
    V37 = np.count_nonzero(Vb.flatten() == 37)*100/Vtotal
    V40 = np.count_nonzero(Vb.flatten() == 40)*100/Vtotal
    V41 = np.count_nonzero(Vb.flatten() == 41)*100/Vtotal
    Vertex_type_list = [V1,V2,V3,V4,V10,V11,V20,V21,V22,V23,V30,V31,V32,V33,V34,V35,V36,V37,V40,V41]
    return (Vertex_type_list,Vb,V) # Vertex percentages
#####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

####Calculate different energies===============================================================
def total_dipolarenergy(IndexArray_non_zeros,size_0Sx,Sxy):#Calculate total dipolar energy of Sxy
    e_d = 0
    Ed_tot = 0
    for index_i in range(0,size_0Sx):            
        yi, xi = IndexArray_non_zeros[index_i][1], IndexArray_non_zeros[index_i][2]
        mi = Sxy[:,yi,xi]
    #    Edi = 0
        e_d = 0
        for index_j in range(0,size_0Sx):            
            yj, xj = IndexArray_non_zeros[index_j][1], IndexArray_non_zeros[index_j][2]
            mj = Sxy[:,yj,xj]
            rij = (xj-xi),(yi-yj) #xdirection is okay, y-direction is reversed
            if (xi != xj or yi!=yj):
                e_d = np.dot(mi,mj)/(np.linalg.norm(rij))**(3) - 3*np.dot(mi,rij)*np.dot(mj,rij)/(np.linalg.norm(rij))**(5)
    #            Edi = Edi + e_d
                Ed_tot = Ed_tot + e_d
    #            print(xi,yi,xj,yj,e_d)
    #    Edxy[yi,xi] = Edi
    return (Ed_tot/2)

### Calculate total energy of single moment in the array=================
def totalenergy_mi(Sxy,xi,yi,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag):
    e_d = 0
    Edi = 0
    Ezi = 0
    Eebi= 0
    mi = Sxy[:,yi,xi]
    Ezi = -1*np.dot(mi,H)  # Zeeman energy
    if (Exchange_Bias==1): # Exchange bias 
        Eebi = -1*np.dot(mi, Hxy_eb_arr[:,yi,xi])
    else:
        Eebi = 0
    e_d = 0
    for index_j in range(0,size_0Sx): #Dipolar energy            
        yj, xj = IndexArray_non_zeros[index_j][1], IndexArray_non_zeros[index_j][2]
        mj = Sxy[:,yj,xj]
        rij = (xj-xi),(yi-yj) #xdirection is okay, y-direction is reversed
        if (xi != xj or yi!=yj):
            e_d = np.dot(mi,mj)/(np.linalg.norm(rij))**(3) - 3*np.dot(mi,rij)*np.dot(mj,rij)/(np.linalg.norm(rij))**(5)
            Edi = Edi + e_d
    Ei_tot = mag*Ezi + mag*Eebi + C*Edi
    return(Ei_tot,C*Edi, mag*Ezi, mag*Eebi)

####=====Calculate total energy of Sxy==============================================================
def totalenergy(Sxy,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag):
    e_d = 0
    Ed_tot = 0
    Ez_tot = 0
    Eeb_tot = 0
    Ez = 0
    for index_i in range(0,size_0Sx):            
        yi, xi = IndexArray_non_zeros[index_i][1], IndexArray_non_zeros[index_i][2]
        mi = Sxy[:,yi,xi]
        Ez = -1*np.dot(mi,H)  # Zeeman energy
        Ez_tot = Ez_tot + Ez 
        if (Exchange_Bias==1): # Exchange bias 
            Eeb = -1*np.dot(mi, Hxy_eb_arr[:,yi,xi])
        else:
            Eeb = 0
        Eeb_tot = Eeb_tot + Eeb
        e_d = 0
        for index_j in range(0,size_0Sx):            
            yj, xj = IndexArray_non_zeros[index_j][1], IndexArray_non_zeros[index_j][2]
            mj = Sxy[:,yj,xj]
            rij = (xj-xi),(yi-yj) #xdirection is okay, y-direction is reversed
            if (xi != xj or yi!=yj):
                e_d = np.dot(mi,mj)/(np.linalg.norm(rij))**(3) - 3*np.dot(mi,rij)*np.dot(mj,rij)/(np.linalg.norm(rij))**(5)
                Ed_tot = Ed_tot + e_d
    E_tot = mag*Ez_tot + mag*Eeb_tot + C*Ed_tot/2
    return(E_tot)
#####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

def EnergyBarrier_arr(Sxy,mag,hb,dhb): #hb:magntitude of barrier and dhb is spread in barrier 
    M,N = np.shape(Sxy)[1], np.shape(Sxy)[2]
    Sx, Sy = Sxy[0,:,:], Sxy[1,:,:]    
    Gaussian_distribution = (mag*hb)*np.random.normal(1,dhb, size=(M*N,)) # Energy barrier distribution (mean, widht, array dimension)
    switching_Barrier = np.reshape(Gaussian_distribution, (M, N)) # Energy barrier distribution
    Ebxy = np.multiply(switching_Barrier,abs(Sx+Sy))   
    return (Ebxy)

def Exchange_bias_arr(Sxy,h_eb,dh_eb, phi_Heb, dphi_Heb):
    M,N = np.shape(Sxy)[1], np.shape(Sxy)[2]
    Sx, Sy = Sxy[0,:,:], Sxy[1,:,:]    
    phi_Heb = 0 # Angle in degrees
    dphi_Heb = 0.0 # Standard deviation in direction of exchange bias
    Exchange_Barrier = h_eb*np.reshape(np.random.normal(1,dh_eb, size=(M*N,)), (M, N)) # Exchange bias distribution (mean, widht, array dimension)
    h_eb_arr = np.multiply(Exchange_Barrier,abs(Sx+Sy))
    Exchange_angle_arr = np.reshape(math.radians(phi_Heb)*np.random.normal(1, dphi_Heb, size=(M*N,)), (M, N))
    Hxy_eb_arr = np.stack([h_eb_arr*np.cos(Exchange_angle_arr),h_eb_arr*np.sin(Exchange_angle_arr)], axis=0)

    hASI = 1
#    if(hASI==1):
#        Hxy_eb_arr[0,1::2,::2] = 0
#        Hxy_eb_arr[0,2::4,3::4] = 0
#        Hxy_eb_arr[0,2::4,::] = 0
#        Hxy_eb_arr[0,0::4,3::4] = 0
#        Hxy_eb_arr[0,::,::] = 0
#        Hxy_eb_arr[0,::6,1::6] = 0.01
        
    return(Hxy_eb_arr)


def MCsimulation(MC_steps,MC_cycles,M,N,IS,hb,dhb,H,Exchange_Bias,h_eb,dh_eb,phi_Heb, dphi_Heb,T,C,mag):
    S, Sxy, Sx, Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    NSx = (N-1)*(M+1)/4 # Total number of spins in Y-sublattice
    NSy = (M-1)*(N+1)/4 # Total number of spins in X-sublattice
    NS  = NSx+NSy # Total number of spins
    IndexArray_non_zeros = np.argwhere(Sxy != 0)
    size_0Sx = np.shape(IndexArray_non_zeros)[0]
    kB =  1.38064852e-23 # Boltzman constant
#    C = 2.8997e-19 # in Joules 1  #1e-7 # Dipolar interaction constant = mu/4pi*mag**2/r**3; r = 340 nm
#    mag = 3.376e-16 #Magnetic moment in Am2 
    Vertex_type_count_arr = np.zeros((1,MC_cycles,20))
    E_tot = 0
    E_MC = []
    E2_MC = []
    
    for j in range(0,MC_cycles):
        S, Sxy, Sx, Sy = ASI_Lattice(M,N,IS)
        Ebxy = EnergyBarrier_arr(Sxy,mag,hb,dhb)
        Hxy_eb_arr = Exchange_bias_arr(Sxy,h_eb,dh_eb, phi_Heb, dphi_Heb)
        E_tot = totalenergy(Sxy,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag)
        Ei = 0
        for i in range (1,MC_steps):  
            axis,yi,xi = random.choice(IndexArray_non_zeros) # choose randomly from the MxN lattice the position of a spin
            Ei,NA1,NA2,NA3 = totalenergy_mi(Sxy,xi,yi,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag)
            dEi = -2*Ei
            dE = dEi + Ebxy[yi,xi] # change in energy if spin mi is flipped
            Prob = np.exp(-Ebxy[yi,xi]/(kB*T))
            Probi = np.exp(-dE/(kB*T))
            rand = random.uniform(0, 1)
            if (dEi<0): # Check for acceptance of flip; flip if change is negative
                if(Prob > rand):
                    E_tot = E_tot+dEi          
                    Sxy[:,yi,xi] = -Sxy[:,yi,xi]
            else: # if dEi is positive
                if(Probi > rand): #flip it
                    E_tot = E_tot+dEi
                    Sxy[:,yi,xi] = -Sxy[:,yi,xi]
    #            else: # do not flip
    #                E_tot = E_tot+0
    #        E_it.append((E_tot))#Append energy list per MC step       
        E_MC.append(E_tot)
        E2_MC.append(E_tot**2)
        Vertex_type_count_arr[0,j,:], V, Vb = Vertex_count_Sxy(Sxy)     
    
    Eav = np.mean(E_MC)/NS #Average energy per spin over (MC_steps-cutoff) steps
    E2av = np.mean(E2_MC)/NS**2 #Average energy per spin over (MC_steps-cutoff) steps
    C_T = (E2av-Eav**2)/(kB*T**2) # Cv at each T        
    Mx=np.sum(Sxy[0,:,:])/NSx # Mx list at temperature T
    My=np.sum(Sxy[1,:,:])/NSy # My list at temperature T
    VertexType_count =  np.mean(Vertex_type_count_arr,axis = 1)
    VertexType_std_count =  np.std(Vertex_type_count_arr,axis = 1)
    return(Eav,C_T,Mx,My,VertexType_count,VertexType_std_count)

def MC_onecyle_simulation(MC_steps,M,N,IS,hb,dhb,H,Exchange_Bias,h_eb,dh_eb,phi_Heb, dphi_Heb,T,C,mag):
    S, Sxy, Sx, Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    NSx = (N-1)*(M+1)/4 # Total number of spins in Y-sublattice
    NSy = (M-1)*(N+1)/4 # Total number of spins in X-sublattice
    NS  = NSx+NSy # Total number of spins
    IndexArray_non_zeros = np.argwhere(Sxy != 0)
    size_0Sx = np.shape(IndexArray_non_zeros)[0]
    kB =  1.38064852e-23 # Boltzman constant
    E_tot = 0
    S, Sxy, Sx, Sy = ASI_Lattice(M,N,IS)
    Ebxy = EnergyBarrier_arr(Sxy,mag,hb,dhb)
    Hxy_eb_arr = Exchange_bias_arr(Sxy,h_eb,dh_eb, phi_Heb, dphi_Heb)
    E_tot = totalenergy(Sxy,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag)
    Ei = 0
    ITERATION = [] # np.zeros(its) # iterations
    ITERATION.append(0)
    E_it = []
    E_it.append(E_tot/NS)
    for i in range (1,MC_steps):  
        axis,yi,xi = random.choice(IndexArray_non_zeros) # choose randomly from the MxN lattice the position of a spin
        Ei,NA1,NA2,NA3 = totalenergy_mi(Sxy,xi,yi,IndexArray_non_zeros,size_0Sx,H,Hxy_eb_arr,Exchange_Bias,C,mag)
        dEi = -2*Ei
        dE = dEi + Ebxy[yi,xi] # change in energy if spin mi is flipped
        Prob = np.exp(-Ebxy[yi,xi]/(kB*T))
        Probi = np.exp(-dE/(kB*T))
        rand = random.uniform(0, 1)
        if (dEi<0): # Check for acceptance of flip; flip if change is negative
            if(Prob > rand):
                E_tot = E_tot+dEi          
                Sxy[:,yi,xi] = -Sxy[:,yi,xi]
        else: # if dEi is positive
            if(Probi > rand): #flip it
                E_tot = E_tot+dEi
                Sxy[:,yi,xi] = -Sxy[:,yi,xi]
#            else: # do not flip
#                E_tot = E_tot+0

        E_it.append(E_tot/NS)#Append energy list per MC step       
        ITERATION.append(i)
    VertexType_count, V, Vb = Vertex_count_Sxy(Sxy)     
    Eav = E_tot/NS
    Mx=np.sum(Sxy[0,:,:])/NSx # Mx list at temperature T
    My=np.sum(Sxy[1,:,:])/NSy # My list at temperature T
    return(Eav,Mx,My,VertexType_count,ITERATION,E_it,Sxy)



def createheader(Xcell,Ycell,MC_steps,MC_cycles,IS,mag,lattice_spacing,C,Ha,phi_H,Temperatures,T,Heb_list,dh_eb,phi_Heb,dphi_Heb,hb,dhb):
#    head = [str(Xcell),str(Ycell),str(MC_steps),str(MC_cycles),str(IS),str(mag),str(lattice_spacing),str(Ha),str(phi_H),str(Temperatures),str(Heb_list),str(dh_eb),str(phi_Heb),str(dphi_Heb),'hb,dhb = '+ str(hb,dhb)]
    head = ["# Start header",\
            'Xcell = ' + str(Xcell),\
            'Ycell = ' + str(Ycell),\
            'MC_steps = ' + str(MC_steps),\
            'MC_cycles = '+str(MC_cycles),\
            'IS = '+ str(IS),\
            'Magnetic moment = '+ str(mag)+' A/m2',\
            'lattice_spacing = '+ str(lattice_spacing)+' nm',\
            'Dipolar coupling = '+str(C) + ' J',\
            'Ha = '+ str(Ha)+' T',\
            'phi_H = '+ str(phi_H)+' degree',\
            'Temperatures = ' + str(Temperatures),\
            'T = =' + str(T)+' K',\
            'Heb_list = '+ str(Heb_list),\
            'dh_eb = ' +str(dh_eb)+' x100%',\
            'phi_Heb = '+ str(phi_Heb)+' deg',\
            'dphi_Heb = ' + str(dphi_Heb)+' x100%',\
            'hb = '+ str(hb)+' T',\
            'dhb = '+str(dhb)+' %',\
            "# End header"]
    head = '\n'.join(head)
    return(head)


######=== Display the ASI data file as image =========================================
def DomainImage(Sxy):  # ASI array to MFM_like image
    Sx, Sy = Sxy[0,:,:], Sxy[1,:,:]    
    pX = Image.open('+Xmagnet.png')
    pY = Image.open('+Ymagnet.png')
    nX = Image.open('-Xmagnet.png')
    nY = Image.open('-Ymagnet.png')
    w = 8
    l = 16
    Xnewsize = (l, w)
    Ynewsize = (w, l)
    pX = pX.resize(Xnewsize)
    nX = nX.resize(Xnewsize)
    pY = pY.resize(Ynewsize)
    nY = nY.resize(Ynewsize)
    n0Sx = np.transpose(Sx[~(Sx==0).all(1)])
    n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
    n0Sy = np.transpose(Sy[~(Sy==0).all(1)])
    n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
    M = n0Sx.shape[1]
    N = n0Sy.shape[0]
    blank_image = Image.new('RGB', (M*(w+l)+w,N*(w+l)+w), (128, 128, 128)) #Image.new('RGB', (M*(w+l)+w,N*(w+l)+w), ((int(np.shape(Sxy)[1]-1)/2*(w+l)+w), int((np.shape(Sxy)[0]-1)/2*(w+l)+w), 128))
    for X in range(0,N):
        for Y in range(0,M+1):
            if (n0Sx[Y,X]==1):
               blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = pX)
            else:
               blank_image.paste(nX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = nX)
    for X in range(0,M+1):
        for Y in range(0,N):
            if (n0Sy[Y,X]==1):
                blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = pY)
            else:
                blank_image.paste(nY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = nY)
#    blank_image.show()
    domain_image = blank_image.convert('L')
#    domain_image.save('Domainstate.png')    
    y=np.asarray(domain_image,dtype=np.uint8) #if values still in range 0-255! 
    noise_matrix = y + np.random.normal(0, 10, (M*(w+l)+w,N*(w+l)+w))#ADD GAUSSIAN NOISE TO MATRIX
    noise_image = Image.fromarray(noise_matrix)
    domain_image = noise_image.convert('L')
    return (domain_image) # MFM_like image with gaussian noise


def Vertex_map(Sxy): # Vertex color map convert ASI MFM_like image to vertex color map
    FinalDomainState = DomainImage(Sxy)
    Xcell = int((np.shape(Sxy)[2]-1)/2)
    Ycell = int((np.shape(Sxy)[1]-1)/2)
    Vertex_type_list,Vb,V = Vertex_count_Sxy(Sxy)
    n0V = Vb[~(Vb==0).all(1)]
    n0V = np.transpose((np.transpose(n0V)[~np.all(np.transpose(n0V) == 0, axis=1)])) 
    Square = 20
    open_cv_image = np.array(FinalDomainState)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB) #cv2.COLOR_GRAY2BGR
    for i in range(0,Xcell-1): #60 vertical lines
        for j in range(0,Ycell-1): #horizontal lines 
            if (n0V[j,i] == 10 or n0V[j,i] == 11):
                cv_img = cv2.rectangle(open_cv_image, (int(28+24*i-Square/2), int(28+24*j-Square/2)), (int(28+24*i+Square/2),int(28+24*j+Square/2)), (0,0,255), 1)
            if (n0V[j,i]==20 or n0V[j,i]==21 or n0V[j,i]==22 or n0V[j,i]==23):
                cv_img = cv2.rectangle(open_cv_image, (int(28+24*i-Square/2), int(28+24*j-Square/2)), (int(28+24*i+Square/2),int(28+24*j+Square/2)), (255,0,0), 1)
            if (n0V[j,i]==30 or n0V[j,i]==31 or n0V[j,i]==32 or n0V[j,i]==33 or n0V[j,i]==34 or n0V[j,i]==35 or n0V[j,i]==36 or n0V[j,i]==37):
                cv_img = cv2.rectangle(open_cv_image, (int(28+24*i-Square/2), int(28+24*j-Square/2)), (int(28+24*i+Square/2),int(28+24*j+Square/2)), (0,255,0), 1)
            if (n0V[j,i]==40 or n0V[j,i]==41):
                cv_img = cv2.rectangle(open_cv_image, (int(28+24*i-Square/2), int(28+24*j-Square/2)), (int(28+24*i+Square/2),int(28+24*j+Square/2)), (0,255,255), 1)
    #cv2.imshow('image',cv_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('Vertex_image.png', Vertex_map(Sxy))
    return(cv_img)
#########&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
def Plot_Eav_MCsteps(ITERATION,E_it,save_data_location,T,h_eb):
    time_stamp=str(datetime.datetime.now().year)+str(datetime.datetime.now().month)+str(datetime.datetime.now().day)#+str(datetime.datetime.now().hour)#now.year, now.month, now.day, now.hour, now.minute, now.second
    f = plt.figure(1)
    plt.clf()
    plt.semilogx(ITERATION,E_it)
    plt.autoscale(enable=True, axis='y')
    plt.title('Average energy vs steps', fontsize=14)
    plt.xlabel('MC steps', fontsize=14)
    plt.ylabel('Average energy (units of Joules)', fontsize=14)
    plt.savefig(save_data_location + '/'+'EvsMC@'+str(int(T))+'K@'+'Heb'+str(h_eb)+'Oe_'+time_stamp+'.png')
    time.sleep(5)
    plt.close(f)
    return()
###########&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    ##----Import data from file as 2D matrix---------------------------------------
def MatrixImport(Fileloc):         
    moment_stringlist = []
    with open(Fileloc) as f:
        for line in f:  # \
            moment_stringlist.append(line)
    
    moment = [] # Moment string list to magnetic moment matrix
    for line in moment_stringlist:
        moment = moment + line.strip( '\n' ).split(' ')    
    
    xx = len(moment_stringlist)
    yy = len(moment_stringlist)
    moment = np.reshape(moment, (xx, yy)).astype(np.float)
    return(moment)
