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
#        self.num1 = tkinter.StringVar(value=folder)#Size_Vertex_matrix
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
                
File_locator = App(tkinter.Tk(), "Tkinter and OpenCV")

Img_path = File_locator.result()#[0]
#folder_name = ntpath.basename(File_locator.result()[0])

onlyfiles = [f for f in listdir(Img_path) if isfile(join(Img_path, f))]

for Img_name in onlyfiles:
    if(Img_name.startswith('M')):
        if not os.path.exists(os.path.join(Img_path,Img_name.split('.png')[0])):
            os.makedirs(os.path.join(Img_path,Img_name.split('.png')[0]))
            copyfile(os.path.join(Img_path,Img_name), os.path.join(Img_path,Img_name.split('.png')[0],Img_name))
            
            temp = list(Img_name)
            temp[0] = 'A'
            temp = ''.join(temp)
            if (os.path.isfile(join(Img_path,temp))==True):
                copyfile(os.path.join(Img_path,temp), os.path.join(Img_path,Img_name.split('.png')[0],temp))
                #print(temp)
            
            