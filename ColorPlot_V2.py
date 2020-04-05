# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:47:14 2018

@author: vmpsk
"""

from os import listdir
from os.path import isfile, join
from shutil import copyfile
from pathlib import Path
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
import os

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

class App(object):
    def __init__(self, window, window_title):#, image_path=(foldername+'.png')):
        self.window = window
        self.window.title(window_title)     
        #-------Frames and grid----------------
        self.frame = tkinter.Frame(window)
        self.frame.grid(row=0,column=0, sticky=tkinter.N)        
        #========Buttons=======================
        self.btn_Read_path = tkinter.Button(self.frame, text="MFM image", width=10, command=self.Read_path) # Button that lets the user to draw grid     
        self.btn_Read_path.grid(row=0,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Read_template = tkinter.Button(self.frame, text="Templates", width=10, command=self.Read_template_path) # Button that lets the user to draw grid     
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
folder_path = file_locator.result()#[0]

#filelist = ['Sx_M12_0.txt', 'Sx_M12_10.txt', 'Sx_M12_20.txt', 'Sx_M12_30.txt', 'Sx_M12_40.txt', 'Sx_M12_50.txt']

#folder_path = "C:/Users/vmpsk/UW_Research/MyWork/Experiments/MFM/M9_Fe10_Pt2_12Jul2018/Demag1_analysed/sf"

onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

i=0
for file in onlyfiles:
#    file = "SF_ASI__M21_250.txt"
    SF = MatrixImport(folder_path + "/"+ file)
    plt.interactive(True)
    plt.ion()
    fft_Final = plt.figure(i)
    plt.imshow(abs(SF),interpolation='nearest', extent=[-6,6,-6,6])
    plt.clim(0,2)
    plt.colorbar()
    plt.savefig(folder_path + "/"+ file.strip( '.txt' ))
    i=i+1