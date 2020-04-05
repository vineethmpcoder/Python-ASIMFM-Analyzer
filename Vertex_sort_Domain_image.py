# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:20:49 2018

@author: vmpsk
"""

from PIL import Image
import numpy as np
import matplotlib.image as img
import numpy as np
import random
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
#from copy import copy, deepcopy
#from numpy import unravel_index
from IPython import get_ipython
#from Vertex_type import Vertex_type
from Function_ASI_Energy import ASI_Lattice
from Function_ASI_Energy import sort_vertex_type
from Function_ASI_Energy import asi_energy
from IPython import get_ipython

#V = Vb
n0V = V[~(V==0).all(1)]
n0V = np.transpose((np.transpose(n0V)[~np.all(np.transpose(n0V) == 0, axis=1)])) 
#ASI lattice with square cell number
Mm, Nn = n0V.shape[0]+1, n0V.shape[1]+1 ## NxN >> (N-1)/2x(N-1)/2 square cells >> (N-3)/2 x (N-3)/2 vertex
#S, Sxy, Sx, Sy = ASI_Lattice(2*Mm+1,2*Nn+1,0) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]

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

for X in range(0,Mm-1):
    for Y in range(0,Nn-1):
        if (n0V[Y,X] == 1):
            blank_image.paste(T1R, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T1R)
        if (n0V[Y,X] == 2):
            blank_image.paste(T2B, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T2B)
        if (n0V[Y,X] == 3):
            blank_image.paste(T3G, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T3G)
        if (n0V[Y,X] == 4):
            blank_image.paste(T4O, (round(X*(l+w)+l), round(l + Y*(l+w))), mask = T4O)


           

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


#blank_image.paste(T1R, (16, 16), mask = T1R)
blank_image.show()
#domain_image = blank_image.convert('L')
#domain_image.save('ExampleDomainstate.png')
#cv2.imwrite('ExampleDomainstate.png',domain_image)
#
#domain_image = Image.fromarray(blank_image)
#im.save("ExampleDomainstate.png")

import scipy.misc
scipy.misc.imsave(folder_path +'/'+'Domain_image.png', blank_image)


#ADD GAUSSIAN NOISE TO MATRIX

#y=np.asarray(domain_image,dtype=np.uint8) #if values still in range 0-255! 
#noise_matrix = y + np.random.normal(0, 50, (1024,1024))
#noise_image = Image.fromarray(noise_matrix)
#noise_image.show()

#blank_image.show()
#blank_image.save('Domainstate.png')



VT = [10,11,20,21,22,23,30,31,32,33,34,35,36,37,40,41]
Total_vertices = 0                
file = open( folder_path +'/'+ 'Vertex_statistics_after_Emin_' + folder_name + '.txt','w') 
#file.write('Threshold = '+ str(threshold) + '\n') 
#file.write('Effective_vertex_matrix_size  = ' + str(lattice_M-2) +'x'+str(lattice_N-2) + '\n') 
#file.write('Xcell  = ' +str(lattice_N-1) + '\n') 
#file.write('Ycell  = ' +str(lattice_N-1) + '\n') 

#Xcell = lattice_M-1
#Ycell = lattice_N-1
Vcounts =[]
for i in range(0,16):
    file.write('T'+str(VT[i]) +' = '+ str(np.count_nonzero(Vb.flatten() == VT[i])) + '\n') 
    print('T'+str(VT[i]) +' = '+ str(np.count_nonzero(Vb.flatten() == VT[i])))
    Vcounts.append(np.count_nonzero(Vb.flatten() == VT[i]))
    Total_vertices = Total_vertices + np.count_nonzero(Vb.flatten() == VT[i])     
print('Total vertices = '+ str(Total_vertices))
file.write('Total_T1 = ' + str(sum(Vcounts[0:2])) + '\n')
file.write('Total_T2 = ' + str(sum(Vcounts[2:6])) + '\n')
file.write('Total_T3 = ' + str(sum(Vcounts[6:14])) + '\n')
file.write('Total_T4 = ' + str(sum(Vcounts[14:16])) + '\n')        
file.write('Percentage_T1 = ' + str(100*sum(Vcounts[0:2])/Total_vertices) + '\n')
file.write('Percentage_T2 = ' + str(100*sum(Vcounts[2:6])/Total_vertices) + '\n')
file.write('Percentage_T3 = ' + str(100*sum(Vcounts[6:14])/Total_vertices) + '\n')
file.write('Percentage_T4 = ' + str(100*sum(Vcounts[14:16])/Total_vertices) + '\n')
file.write('Total_vertices = ' + str(Total_vertices) + '\n')
file.write('Average_mx = ' + str(np.mean(n0Sx)) + '\n')
file.write('Average_my = ' + str(np.mean(n0Sy)) + '\n')
for i in range(0,16):
    file.write('Percentage_T'+str(VT[i]) +' = '+ str(100*np.count_nonzero(Vb.flatten() == VT[i])/Total_vertices) + '\n') 
    print('Percentage_T'+str(VT[i]) +' = '+ str(100*np.count_nonzero(Vb.flatten() == VT[i])/Total_vertices))
file.close() 


Square = 14
######===========Highlight vertex types=====================================        
cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
cv_img = cv2.resize(cv_img, (0,0), fx=image_scale, fy=image_scale) 
n0V = Vb[~(Vb==0).all(1)]
n0V = np.transpose((np.transpose(n0V)[~np.all(np.transpose(n0V) == 0, axis=1)])) 

for j in range(0,Size_Vertex_matrix+2): #60 vertical lines
    for i in range(0,Size_Vertex_matrix+2): #horizontal lines
        if(i>0 and i<Size_Vertex_matrix+1 and j>0 and j<Size_Vertex_matrix+1):                    
            if (n0V[j-1,i-1] == 10 or n0V[j-1,i-1] == 11):
                cv_img = cv2.rectangle(cv_img, (int(Cor_matrix[j,i,0]-Square/2), int(Cor_matrix[j,i,1]-Square/2)), (int(Cor_matrix[j,i,0]+Square/2),int(Cor_matrix[j,i,1]+Square/2)), (255,0,0), 1)
            if (n0V[j-1,i-1]==20 or n0V[j-1,i-1]==21 or n0V[j-1,i-1]==22 or n0V[j-1,i-1]==23):
                cv_img = cv2.rectangle(cv_img, (int(Cor_matrix[j,i,0]-Square/2), int(Cor_matrix[j,i,1]-Square/2)), (int(Cor_matrix[j,i,0]+Square/2),int(Cor_matrix[j,i,1]+Square/2)), (0,0,255), 1)
            if (n0V[j-1,i-1]==30 or n0V[j-1,i-1]==31 or n0V[j-1,i-1]==32 or n0V[j-1,i-1]==33 or n0V[j-1,i-1]==34 or n0V[j-1,i-1]==35 or n0V[j-1,i-1]==36 or n0V[j-1,i-1]==37):
                cv_img = cv2.rectangle(cv_img, (int(Cor_matrix[j,i,0]-Square/2), int(Cor_matrix[j,i,1]-Square/2)), (int(Cor_matrix[j,i,0]+Square/2),int(Cor_matrix[j,i,1]+Square/2)), (0,255,0), 1)
            if (n0V[j-1,i-1]==40 or n0V[j-1,i-1]==41):
                cv_img = cv2.rectangle(cv_img, (int(Cor_matrix[j,i,0]-Square/2), int(Cor_matrix[j,i,1]-Square/2)), (int(Cor_matrix[j,i,0]+Square/2),int(Cor_matrix[j,i,1]+Square/2)), (255,255,0), 1)
            
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
#canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
cv2.imwrite('Vertex_image.png', cv_img)
 
















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