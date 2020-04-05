# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:00:12 2019

@author: vmpsk
"""
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from pathlib import Path
import numpy as np
from numpy import unravel_index
from IPython import get_ipython
import cmath
from pylab import *
from scipy.ndimage import measurements
import collections
import matplotlib.pyplot as plt

import os
import tkinter
import PIL.Image, PIL.ImageTk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from os import listdir
from os.path import isfile, join
from shutil import copyfile

from FunctionList_SquareASI import ASI_Lattice

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

vertexstat_folder = file_locator.result()#[0]


#VertexStat_folder_list = ["C:/Users/vmpsk/UW_Research/MyWork/Simulations/InUse_MFM_analysis/testdata/2pin"] 
#VertexStat_folder_list = ['C:/Users/vmpsk/UW_Research/MyWork/Experiments/MFM/M9_Fe10_Pt2_12Jul2018/Demag1_analysed/M12_70']
stat_array = []
full_stat = []
header = []
#Corr_nn = []
#Corr_2nn = []
i=0
#for vertexstat_folder in VertexStat_folder_list:
Vertex_file = 'V_mfm.txt'

if (os.path.isfile(vertexstat_folder + '/' + Vertex_file)==True):
####----Import data from files-------------------------------------------------
    Sx = MatrixImport(vertexstat_folder+'/'+ 'Sx_mfm.txt')
    Sy = MatrixImport(vertexstat_folder+'/'+ 'Sy_mfm.txt')    
    Vertex = MatrixImport(vertexstat_folder+'/'+Vertex_file)

###############################
    M, N = np.shape(Sx)[0], np.shape(Sx)[1]  ## Matrix size representing the ASI array for calculations and representations 
    IS = 2 # Intial state of the ASI
    G1S, G1Sxy, G1Sx, G1Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    G2S, G2Sxy, G2Sx, G2Sy = ASI_Lattice(M,N,IS) # ASI lattice generate function ASI_Lattice(Rows,Colums, Initial state {0:randomn, 1: DPS}) ; #Sxy[:,1,0] #[x=0 or y=1,row, column]
    G2S, G2Sxy, G2Sx, G2Sy = -1*G2S, -1*G2Sxy, -1*G2Sx, -1*G2Sy



    corr1x = np.sum(np.multiply(Sx,G1Sx))/((M-1)*(N-1)/4)
    corr2x = np.sum(np.multiply(Sx,G2Sx))/((M-1)*(N-1)/4)

    corr1y = np.sum(np.multiply(Sy,G1Sy))/((M-1)*(N-1)/4)
    corr2y = np.sum(np.multiply(Sy,G2Sy))/((M-1)*(N-1)/4)


    print(corr1x, corr2x, corr1y, corr2y)



    n0V = np.transpose(Vertex[~(Vertex==0).all(1)])
    n0V = np.transpose(n0V[~(n0V==0).all(1)])
    T10 = 1*(n0V==10)
    T11 = 1*(n0V==11)
    T2  = 1*(n0V==20) + 1*(n0V==21) + 1*(n0V==22) + 1*(n0V==23)

    T1 = T10+T11
#    T1 = T2
    
    #        plt.interactive(True) 
    #        plt.ion()
    #        Random_matrix = plt.figure(1)
    #        plt.imshow(T1, origin='lower', interpolation='nearest')
    #        plt.colorbar()
    
    T1_domains, num = measurements.label(T1) #  counts and labels clusters
    #        plt.interactive(True) 
    #        plt.ion()
    #        Count_matrix = plt.figure(2)
    #        plt.imshow(lw, origin='lower', interpolation='nearest')
    #        plt.colorbar()
    
    area = measurements.sum(T1, T1_domains, index=arange(T1_domains.max() + 1))
    T1_area_map = area[T1_domains]
    im3 = imshow(T1_area_map, origin='lower', interpolation='nearest')
    colorbar()
    title("Clusters by area")
    show()

    flatten_T1_domain = T1_domains.flatten()
    flat_n0T1_domain = flatten_T1_domain[flatten_T1_domain!=0]
    Area_values = list(collections.Counter(flat_n0T1_domain).values())
#    del Area_values[0]
    Area_statistics = collections.Counter(Area_values)
    Area_statistics=sorted(Area_statistics.items())
    domain_size =  np.array([*Area_statistics])[:,0]
    domain_frequency = np.array([*Area_statistics])[:,1]

plt.plot(domain_size, domain_frequency)

max_domain = 100*np.max(domain_size)/np.size(n0V)
print(max_domain)