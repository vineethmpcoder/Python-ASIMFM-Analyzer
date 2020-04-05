# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:33:01 2018

@author: vmpsk
"""

from os import listdir
from os.path import isfile, join
from shutil import copyfile
from pathlib import Path
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
import tkinter
import PIL.Image, PIL.ImageTk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory



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
import cmath
pi = 3.141592
starttime = datetime.datetime.now()

##----Generate array of float values-------------------------------------------
def frange(start, stop, step):
    arr = []
    i = start
    while i < stop:
         arr.append(i)
         i += step
    arr.append(i)     
    return(arr)         

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

#filelist = ['Sx_M12_0.txt', 'Sx_M12_10.txt', 'Sx_M12_20.txt', 'Sx_M12_30.txt', 'Sx_M12_40.txt', 'Sx_M12_50.txt']

class App(object):
    def __init__(self, window, window_title):#, image_path=(foldername+'.png')):
        self.window = window
        self.window.title(window_title)     
        #-------Frames and grid----------------
        self.frame = tkinter.Frame(window)
        self.frame.grid(row=0,column=0, sticky=tkinter.N)        
        #========Buttons=======================
        self.btn_Read_path = tkinter.Button(self.frame, text="Sxy_folder", width=20, command=self.Read_path) # Button that lets the user to draw grid     
        self.btn_Read_path.grid(row=0,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Read_template = tkinter.Button(self.frame, text="Save_folder", width=20, command=self.Read_template_path) # Button that lets the user to draw grid     
        self.btn_Read_template.grid(row=1,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)                    
        self.window.mainloop()
        
    def Read_path(self):
        self.MFM_folder = askdirectory()
        self.Image_path = tkinter.StringVar(value=self.MFM_folder)
        self.label_Read_path     = tkinter.Label(self.frame,text=self.MFM_folder)
        self.label_Read_path.grid(row=0,column=1)
       
    def Read_template_path(self):
        self.Template_folder = askdirectory()
        self.Template_path = tkinter.StringVar(value=self.Template_folder)
        self.label_Read_template_path     = tkinter.Label(self.frame,text=self.Template_folder)
        self.label_Read_template_path.grid(row=1,column=1)

    def result(self):    
        return self.MFM_folder #, self.Template_folder
                
file_locator = App(tkinter.Tk(), "Tkinter and OpenCV")
datafolder_path = file_locator.result()#[0]
save_path = file_locator.result()#[0]

allfiles = [f for f in listdir(datafolder_path) if isfile(join(datafolder_path, f))]
if('Thumbs.db' in allfiles):
    allfiles.remove('Thumbs.db');

list_Sx_files = []
for file in allfiles:
    if(file.split('_')[0]=='Sx'):
        list_Sx_files.append(file)        

spacing = []
for file in list_Sx_files:
    if(len(file.split('_'))== 4):
        spacing.append(file.split('_')[2])
    if(len(file.split('_')) == 3):
        spacing.append(file.split('_')[2].split('.txt')[0])

spacing = sorted(list(map(int, spacing)))

Sx_file_list = []
for sp in spacing:
    for file in list_Sx_files:        
        if (str(sp) == file.split('_')[2]):
            Sx_file_list.append(file)
            break    


####--Reciprocal space---------------------------------------------------------
Qx = frange(-6*pi,6*pi,0.2*pi)
Qy = frange(-6*pi,6*pi,0.2*pi)

for file in Sx_file_list:
    ##-----------------------------------------------------------------------------
    Xmoment_file = file # 'Sx_mfm.txt'
    Ymoment_file = 'Sy' + file.split('Sx')[-1]#'Sy_mfm.txt'
        
    ####----Import data from files-------------------------------------------------
    mx_moment = MatrixImport(datafolder_path+'/'+Xmoment_file)
    n0Sx = mx_moment
#    n0Sx = np.transpose(mx_moment[~(mx_moment==0).all(1)])
#    n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
#    n0Sx = np.delete(n0Sx, (-1), axis=0)
        
    my_moment = MatrixImport(datafolder_path+'/'+Ymoment_file)
    n0Sy = my_moment
#    n0Sy = np.transpose(my_moment[~(my_moment==0).all(1)])
#    n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)]) 
#    n0Sy = np.delete(n0Sy, (-1), axis=1)
    Sxy = np.zeros((n0Sx.shape[0],n0Sx.shape[0],2))
    Sxy[:,:,0] = n0Sx
    Sxy[:,:,1] = n0Sy
    #####-----Initialize values----------------------------------------------------    
    A = np.zeros((len(Qx),len(Qx),2))
    B = np.zeros((len(Qx),len(Qx),2))
#    position_vector = np.zeros((n0Sx.shape[0],n0Sx.shape[0],2))
    a = np.array([0,0])
    b = np.array([0,0])
    S_per = np.array([0,0])
    #####--------------------------------------------------------------------------
    
    ######-----Calculations of magnetic structure factors---------------------------
    for k in range(len(Qx)-1): # x-axis Qx
        qx = Qx[k]
        for l in range(len(Qy)-1): # y-axis Qy
            qy = Qy[l]
            q = np.array([qy,qx])
            if(qx==0 and qy==0):
                unit_q = np.array([0,0])
            else:
                unit_q = np.array([qx,qy])/np.linalg.norm((qx,qy))        

            a = np.array([0,0])
            b = np.array([0,0])
#            S_per = np.array([0,0])
            for y in range(n0Sx.shape[0]): # y-axis
                i = n0Sx.shape[0]-y-1                
                for x in range(n0Sx.shape[1]): # x-axis
                    r = np.array([x,y])/2
                    s = Sxy[i,x,:] # (0,+1) or (0,-1)                    
                    S_per = s - np.dot(unit_q,s)*unit_q                                        
                    a = a + S_per*np.cos(np.dot(q,r))
                    b = b + S_per*np.sin(np.dot(q,r)) 
#                    position_vector[i,x,:] = (x,y)
                    
            A[(len(Qy)-1)-l,k,:] = a
            B[(len(Qy)-1)-l,k,:] = b           
   
#    plt.interactive(True)
#    plt.ion()
#    fft_Final = plt.figure(3)
#    plt.imshow(G1+G2-(P1+P2), interpolation='nearest')
#    plt.colorbar()
            
    Iq = (A**2 + B**2)/(n0Sx.shape[0]*n0Sx.shape[0]/4)    
    np.savetxt(save_path + '/'+'SF_Sxy_'+file.split('Sx')[-1], Iq[:,:,0]+Iq[:,:,1])
#    np.savetxt(save_path + '/'+'SF_Sx_'+file.split('A')[-1], )
#    np.savetxt(save_path + '/'+'SF_Sy_'+file.split('B')[-1], )

finaltime = datetime.datetime.now()
time_taken = finaltime - starttime
print('Time taken = ', time_taken)
