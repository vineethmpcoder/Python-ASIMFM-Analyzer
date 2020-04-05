# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:14:47 2017

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
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')



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
            
#===============Different intial states
            
#Intialize the random ASI spin array
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
###########################################################################################################################################





def sort_vertex_type(yii,xii,Sxy,M,N): ## *&* Index of the vertex and not of spin
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


def sort_vertex_btype(yii,xii,Sxy,M,N): ## *&* Index of the vertex and not of spin
    Vbtype = 0
    if (1<xii<M-2 and 1<yii<N-2):
        Cl = Sxy[0,yii,xii-1]
        Cr = Sxy[0,yii,xii+1]
        Cd = Sxy[1,yii+1,xii]
        Ct = Sxy[1,yii-1,xii]        
        if     ((abs(Cl) == 1 and abs(Cr) == 1))\
            and ((abs(Cd) == 1 and abs(Ct) == 1)):                
            if (Cl+Cr+Ct+Cd == 2) or (Cl+Cr+Ct+Cd == -2):
                Vtype = 3
                if (Cl==1 and Ct==1 and Cr==1 and Cd==-1):
                    Vbtype = 30
                if (Cl==1 and Ct==-1 and Cr==1 and Cd==1):
                    Vbtype = 31
                if (Cl==-1 and Ct==1 and Cr==1 and Cd==1):
                    Vbtype = 32                    
                if (Cl==1 and Ct==1 and Cr==-1 and Cd==1):
                    Vbtype = 33                    
                if (Cl==-1 and Ct==-1 and Cr==1 and Cd==-1):
                    Vbtype = 34                    
                if (Cl==1 and Ct==-1 and Cr==-1 and Cd==-1):
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
    return (Vbtype)
#####&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
def Sxy_to_Vxy(Sxy):    
    Vxy = np.zeros((Sxy.shape[1],Sxy.shape[2]))
    for xi in range (0,Sxy.shape[1]-1):
        for yi in range (0,Sxy.shape[2]-1):
            if (Sxy[0,yi,xi] != 0): # x compoment
                if (0<xi<Sxy.shape[1]-1):
                    vert_right = xi+1
                    vert_left  = xi-1
                    Vxy[yi,xi+1] = sort_vertex_btype(yi,vert_right,Sxy,Sxy.shape[1],Sxy.shape[2])                
                    Vxy[yi,xi-1] = sort_vertex_btype(yi,vert_left,Sxy,Sxy.shape[1],Sxy.shape[2])
    
            if (Sxy[1,yi,xi] != 0):
                if (0<yi<Sxy.shape[2]-1):
                    vert_top = yi-1
                    vert_bot = yi+1
                    Vxy[yi+1,xi] = sort_vertex_btype(vert_bot,xi,Sxy,Sxy.shape[1],Sxy.shape[2])
                    Vxy[yi-1,xi] = sort_vertex_btype(vert_top,xi,Sxy,Sxy.shape[1],Sxy.shape[2]) 
    return (Vxy)

def GroundState(M,N):
    IS = 2 # Intial state of the ASI
    G1S, G1Sxy, G1Sx, G1Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    G2S, G2Sxy, G2Sx, G2Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    G2S, G2Sxy, G2Sx, G2Sy = -1*G1S, -1*G1Sxy, -1*G1Sx, -1*G1Sy
    V1xy = Sxy_to_Vxy(G1Sxy)
    V2xy = Sxy_to_Vxy(G2Sxy)
    n0V1 = V1xy[~(V1xy==0).all(1)]
    n0V1 = np.transpose((np.transpose(n0V1)[~np.all(np.transpose(n0V1) == 0, axis=1)])) 
    n0V2 = V2xy[~(V2xy==0).all(1)]
    n0V2 = np.transpose((np.transpose(n0V2)[~np.all(np.transpose(n0V2) == 0, axis=1)])) 
    return (n0V1, n0V2)




def asi_energy(xi,yi,Sxy,M,N,H,C, dipolar_switch, exchange_switch, Exchange_Bias, Hxy_eb_arr):
    E, Ed, Ed1, Ed2, Ed3, Ed4, EdT, EdL, EdR, EdB  = 0,0,0,0,0,0,0,0,0,0
    Ex = 0
    Ez = 0
    Eeb = 0
#    R = (yi,xi)  #Position vector of spin "M"
    mi = Sxy[:,yi,xi] 
    if  np.dot(mi,mi)>0: # np.dot(mi,mi)!=0 or
        Ez = -np.dot(mi,H) # Zeeman energy  # Calculate the energy of the single chosen spin

        if exchange_switch == 1: # Exchange interactions with 4 nearest neighbour
            if yi != 0:
                ExT = -np.dot(mi, Sxy[:,yi-2,xi]) # top
            else:
                ExT = 0
            if xi != 0:
                ExL = -np.dot(mi, Sxy[:,yi,xi-2]) # left
            else:
                ExL = 0
            if yi != M-1:
                ExB = -np.dot(mi, Sxy[:,yi+2,xi]) # bottom
            else:
                ExB = 0
            if xi != N-1:
                ExR = -np.dot(mi, Sxy[:,yi,xi+2]) # right
            else:
                ExR = 0                
            Ex = ExL + ExR + ExT + ExB        
    
#=====Calculate dipolar energy of spin and spin-flipped:: Next nearest neigbour========================                                  
        if dipolar_switch == 1:                
            #==========4 Next nearest neighbour dipolar coupling=========================
            if (yi > 1): # mi= (mx,my); mx = Sxy[0,yi,xi] and my = Sx[1,yi,xi]; r = (x,y)
                EdT = C*np.dot(Sxy[:,yi-2,xi],mi)/8  - 3*C*np.dot(Sxy[:,yi-2,xi],(0,2))*np.dot(mi,(0,2))/32
            else:
                EdT = 0
            if (xi > 1):
                EdL = C*np.dot(Sxy[:,yi,xi-2],mi)/8  - 3*C*np.dot(Sxy[:,yi,xi-2],(-2,0))*np.dot(mi,(-2,0))/32
            else:
                EdL = 0
            if (yi < M-2):
                EdB = C*np.dot(Sxy[:,yi+2,xi],mi)/8  - 3*C*np.dot(Sxy[:,yi+2,xi],(0,-2))*np.dot(mi,(0,-2))/32
            else:
                EdB = 0
            if (xi < N-2):
                EdR = C*np.dot(Sxy[:,yi,xi+2],mi)/8  - 3*C*np.dot(Sxy[:,yi,xi+2],(2,0))*np.dot(mi,(2,0))/32
            else:
                EdR = 0

#======Nearest neigbour r = (1,-1),(-1,1),(-1,-1) and (1,1)
            if (yi > 0 and xi < N-1):# mi= (mx,my); mx = Sxy[0,yi,xi] and my = Sx[1,yi,xi]
                Ed1 = -3*C*np.dot(Sxy[:,yi-1,xi+1],(1,1))*np.dot(mi,(1,1))/5.656854#2**(2.5)# + C*np.dot(Sxy[:,yi-1,xi+1],mi)/2**(1.5) # top right
            else:
                Ed1 = 0    # top right
            if (yi > 0 and xi > 0):
                Ed4 = -3*C*np.dot(Sxy[:,yi-1,xi-1],(-1,1))*np.dot(mi,(-1,1))/5.656854#2**(2.5)# + C*np.dot(Sxy[:,yi-1,xi-1],mi)/2**(1.5)
            else:
                Ed4 = 0   # top left
            if (yi < N-1 and xi > 0):
                Ed3 = -3*C*np.dot(Sxy[:,yi+1,xi-1],(-1,-1))*np.dot(mi,(-1,-1))/5.656854#2**(2.5)# + C*np.dot(Sxy[:,yi+1,xi-1],mi)/2**(1.5)
            else:
                Ed3 = 0   # left bottom
            if (yi < N-1 and xi < M-1):
                Ed2 = -3*C*np.dot(Sxy[:,yi+1,xi+1],(1,-1))*np.dot(mi,(1,-1))/5.656854#2**(2.5)# + C*np.dot(Sxy[:,yi+1,xi+1],mi)/2**(1.5)
            else:
                Ed2 = 0    # right bottom    
                        
            Ed = Ed1 + Ed2 + Ed3 + Ed4 # + EdL + EdR + EdT + EdB 
    
    if (Exchange_Bias==1):
        Eeb = -1*np.dot(mi, Hxy_eb_arr[:,yi,xi])
    else:
        Eeb = 0
#===========================================================================================
    E = Ed + Ez + Ex + Eeb
    return(E, Ed, Ez, Ex, Eeb) #+ Ex # Total energy