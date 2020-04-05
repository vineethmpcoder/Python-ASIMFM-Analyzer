# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:32:26 2019

@author: vmpsk
"""


from os import listdir
from os.path import isfile, join
from shutil import copyfile
from pathlib import Path
import numpy as np
from tkinter import Tk

import os
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
#
#class App(object):
#    def __init__(self, window, window_title):#, image_path=(foldername+'.png')):
#        self.window = window
#        self.window.title(window_title)     
#        #-------Frames and grid----------------
#        self.frame = tkinter.Frame(window)
#        self.frame.grid(row=0,column=0, sticky=tkinter.N)        
#        #========Buttons=======================
#        self.btn_Read_path1 = tkinter.Button(self.frame, text="Vertex Stat file select", width=25, command=self.Read_path1) # Button that lets the user to draw grid     
#        self.btn_Read_path1.grid(row=0,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
##        self.btn_Read_path2 = tkinter.Button(self.frame, text="Save Result", width=25, command=self.Read_path2) # Button that lets the user to draw grid     
##        self.btn_Read_path2.grid(row=1,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)                    
##        self.btn_Read_path3 = tkinter.Button(self.frame, text="Read parameters", width=20, command=self.Read_paramters) # Button that lets the user to draw grid     
##        self.btn_Read_path3.grid(row=2,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)                    
#        self.Vfilelist = []
#        self.Parameter = []
#        self.name = tkinter.StringVar(value='')
#        self.File_Labels = []
#        self.name = []
#        
#        self.window.mainloop()
#        
#    def Read_path1(self):
#        self.Path1 = askdirectory()
#        self.Vfilelist.append(self.Path1)
#        fileselected = []
#        self.ii = 0
#        self.File_Labels = []
#        for self.Vfile in self.Vfilelist:
#            fileselected.append(tkinter.Label(self.frame, text=self.Vfile)) #entry.place(x=10, y=10, width=100)
#            fileselected[self.ii].grid(row=3+self.ii, column=0)
#
#            self.File_Labels.append(tkinter.Entry(self.frame,bd=1))#, textvariable=self.name,width=5)) #entry.place(x=10, y=10, width=100)
#            self.File_Labels[self.ii].grid(row=3+self.ii, column=1)
##            print(len(self.File_Labels))
#            self.ii = self.ii + 1            
#              
#    def Read_path2(self):
#        self.Path2 = askdirectory()
#        self.label_Read_path2 = tkinter.Label(self.frame,text=self.Path2)
#        self.label_Read_path2.grid(row=3+50+1,column=0)
# 
#    def Read_paramters(self):
#        self.Parameter = []
#        self.Parameter = [e.get() for e in self.File_Labels]
#        print(self.Parameter)
#               
#    def result(self):
#        return self.Vfilelist#, self.Path2, self.Parameter
#                
#folder_locator = App(tkinter.Tk(), "Tkinter and OpenCV")
##Save_folder_path = folder_locator.result()[0]
#VertexStat_folder_list = folder_locator.result()[0]
##Parameter = folder_locator.result()[2]


VertexStat_folder_list = ["C:/Users/vmpsk/UW_Research/MyWork/Simulations/InUse_MFM_analysis/testdata/2pin"] 
stat_array = []
full_stat = []
header = []
Corr_nn = []
Corr_2nn = []
i=0
for vertexstat_folder in VertexStat_folder_list:
    Xmoment_file = 'Sx_mfm.txt'
    Ymoment_file = 'Sy_mfm.txt' #'Sy' + file.split('Sx')[-1]#'Sy_mfm.txt'

    if (os.path.isfile(vertexstat_folder + '/' + Xmoment_file)==True):
        ####----Import data from files-------------------------------------------------
        mx_moment = MatrixImport(vertexstat_folder+'/'+Xmoment_file)
        n0Sx = mx_moment
        my_moment = MatrixImport(vertexstat_folder+'/'+Ymoment_file)
        n0Sy = my_moment
#        Sxy = np.zeros((n0Sx.shape[0],n0Sx.shape[0],2))
#        Sxy[:,:,0] = n0Sx
#        Sxy[:,:,1] = n0Sy

        n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
        n0Sx = np.transpose(n0Sx[~(n0Sx==0).all(1)])
        n0Sx = np.delete(n0Sx, (-1), axis=0)
        n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
        n0Sy = np.transpose(n0Sy[~(n0Sy==0).all(1)])
        n0Sy = np.delete(n0Sy, (-1), axis=1)            
        Sxy = np.zeros((n0Sx.shape[0],n0Sx.shape[0],2))
        Sxy[:,:,0] = n0Sx
        Sxy[:,:,1] = n0Sy
        #####-----Initialize values----------------------------------------------------    
        a = np.array([0,0])
        b = np.array([0,0])
        #####--------------------------------------------------------------------------
        ####--List of x and y values range---------------------------------------------------------
        X = frange(0,10,1)
        Y = frange(0,10,1)
        r = []
        list_tuple = []
        for x in X:
            for y in Y:
                a = np.sqrt(x**2 + y**2)
                if (a not in r):
                    r.append(np.sqrt(x**2 + y**2))
                    r = sorted(r)
                    t = [x,y]
                    list_tuple.append(t)

        Cxy = np.zeros((n0Sx.shape[0],n0Sx.shape[0],len(r)))
        count = np.zeros((len(r)))
        count_i = 0
        
        for neighbor in list_tuple:
            [x,y] = neighbor
            print(x,y)
            a = np.sqrt(x**2 + y**2)
            distance_index = r.index(a)
            for i in range(n0Sx.shape[0]):
                for j in range(n0Sx.shape[1]):
                    s = Sxy[i,j,0]

                    if a != 0 and s!=0 and i-int(a) >=0 and j-int(a) >=0 and i+int(a) <=n0Sx.shape[0]-2 and j+int(a) <= n0Sx.shape[0]-2 :
                        count[distance_index] = count[distance_index] + 1
                        if((x+y)%2 == 0):
                            if (Sxy[i,j,0] == Sxy[i-y,j-x,0]):
                                Clt = 1
                            else:
                                Clt = -1

                            if (Sxy[i,j,0] == Sxy[i+y,j-x,0]):
                                Clb = 1
                            else:
                                Clb = -1

                            if (Sxy[i,j,0] == Sxy[i-y,j+x,0]):
                                Crt = 1
                            else:
                                Crt = -1
                                
                            if (Sxy[i,j,0] == Sxy[i+y,j+x,0]):
                                Crb = 1
                            else:
                                Crb = -1

                            Cijxy = (Clt + Clb + Crb + Crt)/4

                        if((x+y)%2 != 0):
                            if (Sxy[i,j,0] == Sxy[i-y,j-x,0]):
                                Clt = -1
                            else:
                                Clt = 1

                            if (Sxy[i,j,0] == Sxy[i+y,j-x,0]):
                                Clb = -1
                            else:
                                Clb = 1

                            if (Sxy[i,j,0] == Sxy[i-y,j+x,0]):
                                Crt = -1
                            else:
                                Crt = 1
                                
                            if (Sxy[i,j,0] == Sxy[i+y,j+x,0]):
                                Crb = -1
                            else:
                                Crb = 1
                    
                            Cijxy = (Clt + Clb + Crb + Crt)/4
####################################################################################################
                    if s!=0 and i-int(a) >=0 and j-int(a) >=0 and i+int(a) <=n0Sx.shape[0]-2 and j+int(a) <= n0Sx.shape[0]-2 :

                        
                        if((x+y)%2 == 0):
                            if (Sxy[i,j,0] == Sxy[i-x,j-y,0]):
                                Clt = 1
                            else:
                                Clt = -1

                            if (Sxy[i,j,0] == Sxy[i+x,j-y,0]):
                                Clb = 1
                            else:
                                Clb = -1

                            if (Sxy[i,j,0] == Sxy[i-x,j+y,0]):
                                Crt = 1
                            else:
                                Crt = -1
                                
                            if (Sxy[i,j,0] == Sxy[i+x,j+y,0]):
                                Crb = 1
                            else:
                                Crb = -1

                            Cijyx = (Clt + Clb + Crb + Crt)/4

                        if((x+y)%2 != 0):
                            if (Sxy[i,j,0] == Sxy[i-x,j-y,0]):
                                Clt = -1
                            else:
                                Clt = 1

                            if (Sxy[i,j,0] == Sxy[i+x,j-y,0]):
                                Clb = -1
                            else:
                                Clb = 1

                            if (Sxy[i,j,0] == Sxy[i-x,j+y,0]):
                                Crt = -1
                            else:
                                Crt = 1
                                
                            if (Sxy[i,j,0] == Sxy[i+x,j+y,0]):
                                Crb = -1
                            else:
                                Crb = 1
                    
                            Cijyx = (Clt + Clb + Crb + Crt)/4
####################################################################################################
                        Cxy[i,j,distance_index] = (Cijxy+Cijyx)/2

Corr = np.zeros((len(r)))                                 
for distance_index in range(np.size(count)):
    if count[distance_index] != 0:
        Corr [distance_index] = Cxy[:,:,distance_index].sum()/count[distance_index]
                                                               
r = np.delete(r, (0), axis=0)
Corr = np.delete(Corr, (0), axis=0)

f = plt.figure(1)
plt.clf()
plt.scatter(r,Corr)
f.show()                    
#        Cnn = Cxy.sum()/((n0Sx != 0).sum()+(n0Sy != 0).sum())        
#    Corr_nn.append(Cnn)        



### save correlation data  ###################
#np.savetxt(Save_folder_path + '/'+'Nearest neighbor correlation.txt', np.transpose(np.vstack([Parameter,Corr_nn])), delimiter='\t',fmt='%s')
