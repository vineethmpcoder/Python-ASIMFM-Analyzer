# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:20:49 2018

@author: vmpsk
"""

from PIL import Image
import numpy as np
from IPython import get_ipython
#import matplotlib.image as img
#import numpy as np
#import random
#import math
#from PIL import Image
#import matplotlib.pyplot as plt
#import time
#from copy import copy, deepcopy
#from numpy import unravel_index
#from Vertex_type import Vertex_type
#from Function_ASI_Energy import ASI_Lattice
#from Function_ASI_Energy import sort_vertex_type
#from Function_ASI_Energy import asi_energy
#from IPython import get_ipython

#ASI lattice with square cell number
#M, N = 4, 4 ## NxN >> (N-1)/2x(N-1)/2 square cells >> (N-3)/2 x (N-3)/2 vertex
#S, Sxy, Sx, Sy = ASI_Lattice(2*M+1,2*N+1,2) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]

def DomainImage(Sx, Sy):    
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
    blank_image = Image.new('RGB', (M*(w+l)+w,N*(w+l)+w), (128, 128, 128))
                
    
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
    return (domain_image)
#ADD GAUSSIAN NOISE TO MATRIX

#y=np.asarray(domain_image,dtype=np.uint8) #if values still in range 0-255! 
#noise_matrix = y + np.random.normal(0, 50, (1024,1024))
#noise_image = Image.fromarray(noise_matrix)
#noise_image.show()

#blank_image.show()
#blank_image.save('Domainstate.png')


def VertexMapDI(n0Vb,n0V1,Sx,Sy):
    Mm, Nn = n0Vb.shape[0]+1, n0Vb.shape[1]+1 ## NxN >> (N-1)/2x(N-1)/2 square cells >> (N-3)/2 x (N-3)/2 vertex
    pX = Image.open('+Xmagnet.png')
    pY = Image.open('+Ymagnet.png')
    nX = Image.open('-Xmagnet.png')
    nY = Image.open('-Ymagnet.png')
    T1R = Image.open('T1R.png')
    T1R = T1R.resize((24,24))
    T2B = Image.open('T2B.png')
    T2B = T2B.resize((24,24))
    T3G = Image.open('T3G.png')
    T3G = T3G.resize((24,24))
    T4O = Image.open('T4O.png')
    T4O = T4O.resize((24,24))

    T1 = Image.open('T1.png')
    T1.putalpha(255)
    T1 = T1.resize((24,24))
    T10 = Image.open('T10.png')
    T10.putalpha(255)
    T10 = T10.resize((24,24))

    T20 = Image.open('T20.png')
    T20.putalpha(255)
    T20 = T20.resize((24,24))
#    T21 = Image.open('T21.png')
#    T21.putalpha(255)
#    T21 = T21.resize((24,24))
    T21 = T20.rotate(-90)
    T23 = T20.rotate(-180)
    T22 = T20.rotate(-270)
    
#    T22 = Image.open('T22.png')
#    T22.putalpha(255)
#    T22 = T22.resize((24,24))
#
#    T23 = Image.open('T23.png')
#    T23.putalpha(255)
#    T23 = T23.resize((24,24))

    T30 = Image.open('T30.png')
    T30.putalpha(255)
    T30 = T30.resize((24,24))
    T34 = T30.rotate(-90)
    T36 = T30.rotate(-180)
    T32 = T30.rotate(-270)

#    T32 = Image.open('T32.png')
#    T32.putalpha(255)
#    T32 = T32.resize((24,24))
#    T34 = Image.open('T34.png')
#    T34.putalpha(255)
#    T34 = T34.resize((24,24))
#    T36 = Image.open('T36.png')
#    T36.putalpha(255)
#    T36 = T36.resize((24,24))

    T4O = Image.open('T4O.png')
    T4O.putalpha(255)
    T4O = T4O.resize((24,24))

    w = 8
    l = 16
    Xnewsize = (l, w)
    Ynewsize = (w, l)
    pX = pX.resize(Xnewsize)
    nX = nX.resize(Xnewsize)
    pY = pY.resize(Ynewsize)
    nY = nY.resize(Ynewsize)
    
    blank_image = Image.new('RGB', (Mm*(w+l)+w,Nn*(w+l)+w), (128, 128, 128, 0))
    Xi=0
    Yi=0
    
    n0Sx = np.transpose(Sx[~(Sx==0).all(1)])
    n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
    n0Sy = np.transpose(Sy[~(Sy==0).all(1)])
    n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
    
    CT = [(255,255,0),(255,0,255),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]#,(0,0,128), (255,0,0), (255,0,0), (255,0,0), (255,0,0)]#,(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0)]
    
    for Y in range(0,Mm+1):
        for X in range(0,Nn):
            if (n0Sx[Y,X]==1):
               blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = pX)
            else:
               blank_image.paste(nX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = nX)
    
    for Y in range(0,Nn):
        for X in range(0,Mm+1):
            if (n0Sy[Y,X]==1):
                blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = pY)
            else:
                blank_image.paste(nY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = nY)
            
    for X in range(0,Mm-1):
        for Y in range(0,Nn-1):
#            if (n0Vb[Y,X] == 10 or n0Vb[Y,X] == 11):
#                blank_image.paste(T1, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T1)

            if (n0Vb[Y,X] == 10 or n0Vb[Y,X] == 11):
                if(n0Vb[Y,X]==n0V1[Y,X]):
                    blank_image.paste(T10, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T10)
                else:
                    blank_image.paste(T1, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T1)


            if (n0Vb[Y,X] == 20):
                blank_image.paste(T20, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T20)
            if (n0Vb[Y,X] == 21):
                blank_image.paste(T21, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T21)                    
            if (n0Vb[Y,X] == 22):
                blank_image.paste(T22, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T22)
            if (n0Vb[Y,X] == 23):
                blank_image.paste(T23, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T23)
                
            if (n0Vb[Y,X] == 30 or n0Vb[Y,X] == 31):
                blank_image.paste(T30, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T30)
            if (n0Vb[Y,X] == 32 or n0Vb[Y,X] == 33):
                blank_image.paste(T32, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T32)
            if (n0Vb[Y,X] == 34 or n0Vb[Y,X] == 35):
                blank_image.paste(T34, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T34)
            if (n0Vb[Y,X] == 36 or n0Vb[Y,X] == 37):
                blank_image.paste(T36, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T36)

            if (n0Vb[Y,X] == 40 or n0Vb[Y,X] == 41):
                blank_image.paste(T4O, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T4O)
#    blank_image.show()
#    VertexMap_DI = blank_image.convert('L')
#    domain_image.save('Domainstate.png')
    return (blank_image)



def VertexMappinDI(n0Vb,n0V1,Sx,Sy,PSx,PSy):
    Mm, Nn = n0Vb.shape[0]+1, n0Vb.shape[1]+1 ## NxN >> (N-1)/2x(N-1)/2 square cells >> (N-3)/2 x (N-3)/2 vertex
    pX = Image.open('+Xmagnet.png')
    pY = Image.open('+Ymagnet.png')
    nX = Image.open('-Xmagnet.png')
    nY = Image.open('-Ymagnet.png')
    T1R = Image.open('T1R.png')
    T1R = T1R.resize((24,24))
    T2B = Image.open('T2B.png')
    T2B = T2B.resize((24,24))
    T3G = Image.open('T3G.png')
    T3G = T3G.resize((24,24))
    T4O = Image.open('T4O.png')
    T4O = T4O.resize((24,24))

    T1 = Image.open('T1.png')
    T1.putalpha(255)
    T1 = T1.resize((24,24))
    T10 = Image.open('T10.png')
    T10.putalpha(255)
    T10 = T10.resize((24,24))

    T20 = Image.open('T20.png')
    T20.putalpha(255)
    T20 = T20.resize((24,24))
#    T21 = Image.open('T21.png')
#    T21.putalpha(255)
#    T21 = T21.resize((24,24))
    T21 = T20.rotate(-90)
    T23 = T20.rotate(-180)
    T22 = T20.rotate(-270)
    
#    T22 = Image.open('T22.png')
#    T22.putalpha(255)
#    T22 = T22.resize((24,24))
#
#    T23 = Image.open('T23.png')
#    T23.putalpha(255)
#    T23 = T23.resize((24,24))

    T30 = Image.open('T30.png')
    T30.putalpha(255)
    T30 = T30.resize((24,24))
    T34 = T30.rotate(-90)
    T36 = T30.rotate(-180)
    T32 = T30.rotate(-270)

#    T32 = Image.open('T32.png')
#    T32.putalpha(255)
#    T32 = T32.resize((24,24))
#    T34 = Image.open('T34.png')
#    T34.putalpha(255)
#    T34 = T34.resize((24,24))
#    T36 = Image.open('T36.png')
#    T36.putalpha(255)
#    T36 = T36.resize((24,24))

    T4O = Image.open('T4O.png')
    T4O.putalpha(255)
    T4O = T4O.resize((24,24))

    w = 8
    l = 16
    Xnewsize = (l, w)
    Ynewsize = (w, l)
    pX = pX.resize(Xnewsize)
    nX = nX.resize(Xnewsize)
    pY = pY.resize(Ynewsize)
    nY = nY.resize(Ynewsize)
    
    blank_image = Image.new('RGB', (Mm*(w+l)+w,Nn*(w+l)+w), (128, 128, 128, 0))
    Xi=0
    Yi=0
    
    n0Sx = np.transpose(Sx[~(Sx==0).all(1)])
    n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
    n0Sy = np.transpose(Sy[~(Sy==0).all(1)])
    n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
    
    CT = [(255,255,0),(255,0,255),(0,0,255),(0,255,0),(255,0,0),(0,255,255)]#,(0,0,128), (255,0,0), (255,0,0), (255,0,0), (255,0,0)]#,(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0),(0,255,0)]
    
    for Y in range(0,Mm+1):
        for X in range(0,Nn):
            if (n0Sx[Y,X]==1):
               blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = pX)
            else:
               blank_image.paste(nX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = nX)
    
    for Y in range(0,Nn):
        for X in range(0,Mm+1):
            if (n0Sy[Y,X]==1):
                blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = pY)
            else:
                blank_image.paste(nY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = nY)
            
    for X in range(0,Mm-1):
        for Y in range(0,Nn-1):
#            if (n0Vb[Y,X] == 10 or n0Vb[Y,X] == 11):
#                blank_image.paste(T1, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T1)

            if (n0Vb[Y,X] == 10 or n0Vb[Y,X] == 11):
                if(n0Vb[Y,X]==n0V1[Y,X]):
                    blank_image.paste(T10, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T10)
                else:
                    blank_image.paste(T1, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T1)

            if (n0Vb[Y,X] == 20):
                blank_image.paste(T20, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T20)
            if (n0Vb[Y,X] == 21):
                blank_image.paste(T21, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T21)                    
            if (n0Vb[Y,X] == 22):
                blank_image.paste(T22, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T22)
            if (n0Vb[Y,X] == 23):
                blank_image.paste(T23, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T23)
            if (n0Vb[Y,X] == 30 or n0Vb[Y,X] == 31):
                blank_image.paste(T30, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T30)
            if (n0Vb[Y,X] == 32 or n0Vb[Y,X] == 33):
                blank_image.paste(T32, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T32)
            if (n0Vb[Y,X] == 34 or n0Vb[Y,X] == 35):
                blank_image.paste(T34, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T34)
            if (n0Vb[Y,X] == 36 or n0Vb[Y,X] == 37):
                blank_image.paste(T36, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T36)
            if (n0Vb[Y,X] == 40 or n0Vb[Y,X] == 41):
                blank_image.paste(T4O, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T4O)

    for Y in range(0,Mm+1):
        for X in range(0,Nn):
            if (PSx[Y,X]==1):
               blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = pX)
#            else:
#               blank_image.paste(nX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = nX)
    
#    for Y in range(0,Nn):
#        for X in range(0,Mm+1):
#            if (PSy[Y,X]==1):
#                blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = pY)
##            else:
#                blank_image.paste(nY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = nY)

#    VertexMap_DI = blank_image.convert('L')
    return (blank_image)




def PottsMappinDI(n0Vb, spin_direction):
    Mm, Nn = n0Vb.shape[0]+1, n0Vb.shape[1]+1 ## NxN >> (N-1)/2x(N-1)/2 square cells >> (N-3)/2 x (N-3)/2 vertex
#    pX = Image.open('+Xmagnet.png')
#    pY = Image.open('+Ymagnet.png')
#    nX = Image.open('-Xmagnet.png')
#    nY = Image.open('-Ymagnet.png')

    if spin_direction == "0":
        spin_img = 'T20.png'
    if spin_direction == "22.5":
        spin_img = 'T225.png'
    if spin_direction == "45":
        spin_img = 'T32.png'
        
#    spin_img = 'T20.png'
    s1 = Image.open(spin_img)
    s1.putalpha(255)
    s1 = s1.resize((24,24))
    s2 = s1.rotate(-90) # clockwise 90 deg
    s3 = s1.rotate(-180) # clockwise 180 deg
    s4 = s1.rotate(-270) # clockwise 270 deg
    
    w = 8
    l = 16
#    Xnewsize = (l, w)
#    Ynewsize = (w, l)
#    pX = pX.resize(Xnewsize)
#    nX = nX.resize(Xnewsize)
#    pY = pY.resize(Ynewsize)
#    nY = nY.resize(Ynewsize)
    blank_image = Image.new('RGB', (Mm*(w+l)+w,Nn*(w+l)+w), (128, 128, 128, 0))
#    Xi=0
#    Yi=0
#    
##    n0Sx = np.transpose(Sx[~(Sx==0).all(1)])
##    n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
##    n0Sy = np.transpose(Sy[~(Sy==0).all(1)])
##    n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
#    
#    for Y in range(0,Mm+1):
#        for X in range(0,Nn):
#            if (n0Sx[Y,X]==1):
#               blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = pX)
#            else:
#               blank_image.paste(nX, (round(l/2 + X*(l+w)), round( Y*(l+w))), mask = nX)
#    
#    for Y in range(0,Nn):
#        for X in range(0,Mm+1):
#            if (n0Sy[Y,X]==1):
#                blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = pY)
#            else:
#                blank_image.paste(nY, (round(X*(l+w)), round(l/2 + Y*(l+w))), mask = nY)
            
    for X in range(0,Mm-1):
        for Y in range(0,Nn-1):
            if (n0Vb[Y,X] == 1):
                blank_image.paste(s1, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = s1)
            if (n0Vb[Y,X] == 2):
                blank_image.paste(s2, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = s2)                    
            if (n0Vb[Y,X] == 3):
                blank_image.paste(s3, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = s3)
            if (n0Vb[Y,X] == 4):
                blank_image.paste(s4, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = s4)
    return (blank_image)












#####----------------test works very well-----------------------------#########
#for Y in range(0,20):
#    for X in range(0,21): 
#        blank_image.paste(pX, (round(l/2 + X*(l+w)), round( Y*(l+w))))
#for Y in range(0,20):
#    for X in range(0,21): 
#        blank_image.paste(pY, (round(X*(l+w)), round(l/2 + Y*(l+w))))












#blank_image.paste(Ymagnet, (0,32)) # (X,Y) positions
#blank_image.paste(Xmagnet, (32,0)) # (X,Y) positions 
#blank_image.paste(Ymagnet, (96,32)) # (X,Y) positions 
#blank_image.paste(Xmagnet, (32,96)) # (X,Y) positions 