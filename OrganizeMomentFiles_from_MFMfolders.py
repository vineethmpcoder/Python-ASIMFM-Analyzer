# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:16:44 2018

@author: vmpsk
"""

import os
import tkinter
import PIL.Image, PIL.ImageTk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from os import listdir
from os.path import isfile, join
from shutil import copyfile

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
#folder_path = 'C:/Users/vmpsk/UW_Research/MyWork/Experiments/MFM/D1_PIC_M10_Fe10_IrMn8_Pt2_BIAS/D50h'
folder_names = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

for folder in folder_names:
    Sx_path=(folder_path + '/' + folder + '/' + 'Sx_mfm'+'.txt')
    Sy_path=(folder_path + '/' + folder + '/' + 'Sy_mfm'+'.txt')
    copyfile(Sx_path, folder_path+'/'+'Sx_'+  folder +'.txt')
    copyfile(Sy_path, folder_path+'/'+'Sy_'+  folder +'.txt')
      
onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

Sx_files = []
Sy_files = []
for name in onlyfiles:
    if name.startswith(('Sx')):
        Sx_files.append(name)
        
    if name.startswith(('Sy')):
        Sy_files.append(name)

####----Sort the file names in ascending order of spacing----------------------
spacings = [] # list of spacings
beforesubstring = [] # list of file name prefix
aftersubstring = []  # list of file name suffix
#### create the lists of file name prefix, spacing and file name suffix
for name in Sx_files:
    if(len(name.split('_'))==4):
        spacings.append(int(name.split('_')[2]))        
        beforesubstring.append(name.split(name.split('_')[2])[0]) #name.split('_')[2]
        aftersubstring.append('_'+ name.split(name.split('_')[2]+'_')[-1])
#        print(name.split(name.split('_')[2])[-1], name)
        
    if(len(name.split('_'))==3):
        spacings.append(int(name.split('_')[2].split('.txt')[0]))
        beforesubstring.append(name.split(name.split('_')[2].split('.txt')[0])[0])
        aftersubstring.append(name.split(name.split('_')[2].split('.txt')[0])[-1])
    
Sorted_spacings = sorted(spacings, key=int)  # sort spacings in ascending o

sortedfile_Sx = []
i = 0
for spacing in Sorted_spacings:
    for a in spacings:
        if (a == spacing):
            sortedfile_Sx.append(beforesubstring[spacings.index(a)]+str(a)+aftersubstring[spacings.index(a)])

            





    