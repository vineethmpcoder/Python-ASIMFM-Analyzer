# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:43:40 2018

"""
import cv2
import numpy as np
#import matplotlib.image as img
import numpy as np
import random
import math
import collections
#from PIL import Image
#import matplotlib.pyplot as plt
#import time
#from IPython import get_ipython
#import psutil
#import turtle
#from PIL import Image, ImageDraw
import tkinter
import PIL.Image, PIL.ImageTk
from PIL import Image
#from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import ntpath
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from pathlib import Path
import os

#from functools import partial
from Function_Display_ASI import DomainImage, VertexMapDI, VertexMappinDI, PottsMappinDI
from Function_ASI_Energy import ASI_Lattice
from Function_ASI_Energy import sort_vertex_btype
from Function_ASI_Energy import asi_energy
from Function_Display_ASI import DomainImage
from Function_ASI_Energy import GroundState
import scipy.misc


class App(object):
    def __init__(self, window, window_title):#, image_path=(foldername+'.png')):
#        self.num1 = tkinter.StringVar(value=folder)#Size_Vertex_matrix
        self.window = window
        self.window.title(window_title)
        
        #-------Frames and grid----------------
        self.frame = tkinter.Frame(window)
        self.frame.grid(row=0,column=0, sticky=tkinter.N)
        
        #========Buttons=======================
        self.btn_Read_path = tkinter.Button(self.frame, text="MFM image", width=40, command=self.Read_path) # Button that lets the user to draw grid     
        self.btn_Read_path.grid(row=0,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_Read_template = tkinter.Button(self.frame, text="Templates", width=40, command=self.Read_template_path) # Button that lets the user to draw grid     
        self.btn_Read_template.grid(row=1,column=0) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)

#        var = tkinter.IntVar()
#        self.color_type = tkinter.Checkbutton(self.frame, text="Type 1", variable=var)
#        self.color_type.grid(row=2,column=0)
#
#        var = tkinter.IntVar()
#        self.color_type = tkinter.Checkbutton(self.frame, text="Type 2", variable=var)
#        self.color_type.grid(row=3,column=0)
#
#        var = tkinter.IntVar()
#        self.color_type = tkinter.Checkbutton(self.frame, text="Type 3", variable=var)
#        self.color_type.grid(row=4,column=0)
#
#        var = tkinter.IntVar()
#        self.color_type = tkinter.Checkbutton(self.frame, text="Type 4", variable=var)
#        self.color_type.grid(row=5,column=0)
                    
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
        return self.MFM_folder, self.Template_folder
                
File_locator = App(tkinter.Tk(), "Tkinter and OpenCV")
folder_name = ntpath.basename(File_locator.result()[0])
template_path = File_locator.result()[0]
folder_path = File_locator.result()[0]
image_path=(File_locator.result()[0] + '/' + folder_name +'.png')

#Cor_matrix =np.zeros((10+1,10+1,2))   
#Size_Vertex_matrix = 0
#X_step_size = 0#37.55 # vertical lines 
#Y_step_size = 0#37.4 # horizontal lines
#X_tilt = 0 # tilt vertical lines +ve countercloswise
#Y_tilt = 0 # tilt horizontal line -ve counter clock
#X_off = 0 #itial position of vertical grid line
#Y_off = 0 #intiial position of horizontal grid line
#X_shift = 0 # position of magnetic element from vertex center
#Y_shift = 0 # position of magnetic element from vertex center
width = 0 #not in use but do not delete
length = 0 #not in use but do not delete
image_scale = 1


class App(object):
    def __init__(self, window, window_title):#, image_path=(foldername+'.png')):
        self.num0 = tkinter.StringVar(value='76')#Size_Vertex_matrix
        self.num1 = tkinter.StringVar(value='12.86')#X_step_size        
        self.num2 = tkinter.StringVar(value='12.91')#Y_step_size
        self.num3 = tkinter.StringVar(value='0.3')#X_tilt
        self.num4 = tkinter.StringVar(value='0.0')#Y_tilt
        self.num5 = tkinter.StringVar(value='14')#X_off
        self.num6 = tkinter.StringVar(value='8')#Y_off
        self.num7 = tkinter.StringVar(value='0')#X_shift
        self.num8 = tkinter.StringVar(value='0')#Y_shift
        self.num9 = tkinter.StringVar(value='13')#length
        self.num10 = tkinter.StringVar(value='8')#width
        self.num11 = tkinter.StringVar(value='12')#bounding square
        self.num12 = tkinter.StringVar(value='0')#location row
        self.num13 = tkinter.StringVar(value='0')#location column
        self.num14 = tkinter.StringVar(value='0')#Pin start row
        self.num15 = tkinter.StringVar(value='0')#Pin start column
        self.num16 = tkinter.StringVar(value='3')#Pinning periodX
        self.num17 = tkinter.StringVar(value='3')#Pinning periodY
        self.num =[self.num0,self.num1,self.num2,self.num3,self.num4,self.num5,
                   self.num6,self.num7,self.num8,self.num9,self.num10,self.num11,
                   self.num12,self.num13,self.num14,self.num15,self.num16,self.num17]
       
        onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        grid_file = ''        
        for file_name in onlyfiles:
            if not(file_name.find('MFM_grid_spec_')==-1):
                grid_file = file_name
                print(grid_file)
            
        if (os.path.isfile(join(folder_path, grid_file))==True):
            num = []
            i=0
            
            with open(join(folder_path, grid_file)) as f:
                for line in f:
                    num.append(line.split('= ')[1].split('\n')[0])
                    self.num[i] = tkinter.StringVar(value=line.split('= ')[1].split('\n')[0])
                    i = i+1
            i=0

        self.window = window
        self.window.title(window_title)

        #####-------Load the MFM image on the canvas--------------------------------------
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale)         
        self.height, self.width, no_channels = self.cv_img.shape # Get the image dimensions (OpenCV stores image data as NumPy ndarray)        
        self.canvas = tkinter.Canvas(window, width = self.width, height = self.height) # Create a canvas that can fit the above image
        self.canvas.grid(row=0,column=1) #        self.canvas.pack()
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img)) # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage       
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW) # Add a PhotoImage to the Canvas        
        self.frame = tkinter.Frame(window)
        self.frame.grid(row=0,column=0, sticky=tkinter.N)

        #######============BUTTONS=====================================================
        self.btn_blur=tkinter.Button(self.frame, text="Update", width=8, command=self.update) # Button that lets the user to draw grid     
        self.btn_blur.grid(row=10,column=1) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_save=tkinter.Button(self.frame, text="Save spec", width=8, command=self.savegrid) # Button that lets the user save the grid       
        self.btn_save.grid(row=55,column=1) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_count = tkinter.Button(self.frame, text="Count", width=8, command=self.count) # Button that lets the user save the grid       
        self.btn_count.grid(row=56,column=1) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_savegrid = tkinter.Button(self.frame, text="Save_grid", width=8, command=self.save_img_grid) # Button that lets the user save the grid       
        self.btn_savegrid.grid(row=57,column=1) #self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)

        ########-------LABELS---------------------
        self.label_Size         = tkinter.Label(self.frame,text='Vertex_matrix')
        self.label_X_step       = tkinter.Label(self.frame,text='Vert line_step')
        self.label_Y_step       = tkinter.Label(self.frame,text='Horz line_step')
        self.label_X_tilt       = tkinter.Label(self.frame,text='VertTilt_+Clo')#.grid(row=1,column=0,sticky=tkinter.E)
        self.label_Y_tilt       = tkinter.Label(self.frame,text='HorzTilt_+Clo')#.grid(row=1,column=0,sticky=tkinter.E)
        self.label_X_off        = tkinter.Label(self.frame,text='Vert_off')#.grid(row=2,column=0,sticky=tkinter.E)
        self.label_Y_off        = tkinter.Label(self.frame,text='Horz_off')#.grid(row=2,column=0,sticky=tkinter.E)
        self.label_X_shift      = tkinter.Label(self.frame,text='X_shift')
        self.label_Y_shift      = tkinter.Label(self.frame,text='Y_shift')
        self.label_m            = tkinter.Label(self.frame,text='m(row)')
        self.label_n            = tkinter.Label(self.frame,text='n(column)')
        self.label_mnwidth      = tkinter.Label(self.frame,text='Width')
        self.label_mnlength     = tkinter.Label(self.frame,text='Length')
        self.label_Square       = tkinter.Label(self.frame,text='Square')
        self.label_PinStartRow  = tkinter.Label(self.frame,text='PinStartRow')
        self.label_PinStartCol  = tkinter.Label(self.frame,text='PinStartCol')
        self.label_PinPeriodX   = tkinter.Label(self.frame,text='PinPeriodX')
        self.label_PinPeriodY   = tkinter.Label(self.frame,text='PinPeriodY')
        ######===LOCATION OF LABELS ON GRID==================
        self.label_Size.grid        (row=11,column=0,sticky=tkinter.E)
        self.label_X_step.grid      (row=12,column=0,sticky=tkinter.E)
        self.label_Y_step.grid      (row=13,column=0,sticky=tkinter.E)
        self.label_X_tilt.grid      (row=14,column=0,sticky=tkinter.E)               
        self.label_Y_tilt.grid      (row=15,column=0,sticky=tkinter.E)
        self.label_X_off.grid       (row=16,column=0,sticky=tkinter.E)
        self.label_Y_off.grid       (row=17,column=0,sticky=tkinter.E)
        self.label_X_shift.grid     (row=18,column=0,sticky=tkinter.E)
        self.label_Y_shift.grid     (row=19,column=0,sticky=tkinter.E)
        self.label_mnlength.grid    (row=20,column=0,sticky=tkinter.E)
        self.label_mnwidth.grid     (row=21,column=0,sticky=tkinter.E)
        self.label_Square.grid      (row=22,column=0,sticky=tkinter.E)
        self.label_m.grid           (row=23,column=0,sticky=tkinter.E)
        self.label_n.grid           (row=24,column=0,sticky=tkinter.E)
        self.label_PinStartRow.grid (row=25,column=0,sticky=tkinter.E)
        self.label_PinStartCol.grid (row=26,column=0,sticky=tkinter.E)
        self.label_PinPeriodX.grid  (row=27,column=0,sticky=tkinter.E)
        self.label_PinPeriodY.grid  (row=28,column=0,sticky=tkinter.E)
        
        ######---------ENTRIES---------------------        
        self.entry_Size          = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[0])  
        self.entry_X_step        = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[1])
        self.entry_Y_step        = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[2])
        self.entry_X_tilt        = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[3])        
        self.entry_Y_tilt        = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[4])        
        self.entry_X_off         = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[5])        
        self.entry_Y_off         = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[6])
        self.entry_X_shift       = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[7])
        self.entry_Y_shift       = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[8])        
        self.entry_mnlength      = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[9])
        self.entry_mnwidth       = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[10])        
        self.entry_Square        = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[11])        
        self.entry_m             = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[12])
        self.entry_n             = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[13])        
        self.entry_PinStartRow   = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[14])
        self.entry_PinStartCol   = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[15])        
        self.entry_PinPeriodX    = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[16])        
        self.entry_PinPeriodY    = tkinter.Entry(self.frame,bd=5,width=10, textvariable=self.num[17])        
        ######===ENTRIES ON GRID==================
        self.entry_Size.grid        (row=11,column=1)
        self.entry_X_step.grid      (row=12,column=1)        
        self.entry_Y_step.grid      (row=13,column=1)
        self.entry_X_tilt.grid      (row=14,column=1)
        self.entry_Y_tilt.grid      (row=15,column=1)
        self.entry_X_off.grid       (row=16,column=1)
        self.entry_Y_off.grid       (row=17,column=1)        
        self.entry_X_shift.grid     (row=18,column=1)
        self.entry_Y_shift.grid     (row=19,column=1)                  
        self.entry_mnlength.grid    (row=20,column=1)
        self.entry_mnwidth.grid     (row=21,column=1)          
        self.entry_Square.grid      (row=22,column=1)          
        self.entry_m.grid           (row=23,column=1)
        self.entry_n.grid           (row=24,column=1)
        self.entry_PinStartRow.grid (row=25,column=1)
        self.entry_PinStartCol.grid (row=26,column=1)
        self.entry_PinPeriodX.grid   (row=27,column=1)
        self.entry_PinPeriodY.grid   (row=28,column=1)
                
        self.colorT1 = tkinter.IntVar()
        self.entry_color_T1 = tkinter.Checkbutton(self.frame, text="S1", variable=self.colorT1)
        self.entry_color_T1.grid(row=11,column=2)        
        self.colorT2 = tkinter.IntVar()
        self.entry_color_T2 = tkinter.Checkbutton(self.frame, text="S2", variable=self.colorT2)
        self.entry_color_T2.grid(row=12,column=2)        
        self.colorT3 = tkinter.IntVar()
        self.entry_color_T3 = tkinter.Checkbutton(self.frame, text="S3", variable=self.colorT3)
        self.entry_color_T3.grid(row=13,column=2)        
        self.colorT4 = tkinter.IntVar()
        self.entry_color_T4 = tkinter.Checkbutton(self.frame, text="S4", variable=self.colorT4)
        self.entry_color_T4.grid(row=14,column=2)
        self.colorPin = tkinter.IntVar()
        self.entry_colorPin = tkinter.Checkbutton(self.frame, text="P", variable=self.colorPin)
        self.entry_colorPin.grid(row=15,column=2)          
        self.colorRevPin = tkinter.IntVar()
        self.entry_colorRevPin = tkinter.Checkbutton(self.frame, text="RP", variable=self.colorRevPin)
        self.entry_colorRevPin.grid(row=16,column=2)            
        
        self.spin_direction = tkinter.StringVar()
        self.spin_direction.set("1") # default value
        self.entry_spin_direction = tkinter.OptionMenu(self.frame, self.spin_direction, "0", "22.5", "45")
        self.entry_spin_direction.grid(row=17,column=2)
        ####==============MAIN===========================================================
        self.window.mainloop()


#######===== Callback for the "UPDATE" button====================================
    def update(self):
        y_start = 0
        x_start = 0
        self.Size_Vertex_matrix = int((self.entry_Size.get()))
        self.X_step_size = float((self.entry_X_step.get()))
        self.Y_step_size = float((self.entry_Y_step.get()))
        self.X_tilt = float((self.entry_X_tilt.get()))
        self.Y_tilt = float((self.entry_Y_tilt.get()))
        self.X_off = int((self.entry_X_off.get()))
        self.Y_off = int((self.entry_Y_off.get()))
        self.X_shift = int((self.entry_X_shift.get()))
        self.Y_shift = int((self.entry_Y_shift.get()))        
        self.m = int((self.entry_m.get()))
        self.n = int((self.entry_n.get())) 
        self.mnlength = int((self.entry_mnlength.get()))
        self.mnwidth = int((self.entry_mnwidth.get()))
        self.Square  = int((self.entry_Square.get())) 
        self.PinStartRow  = int((self.entry_PinStartRow.get())) #PinStartRow + i*self.entry_PinPeriodX
        self.PinStartCol  = int((self.entry_PinStartCol.get()))
        self.PinPeriodX  = int((self.entry_PinPeriodX.get()))
        self.PinPeriodY  = int((self.entry_PinPeriodY.get()))
        
        ####----------Reload the original image again------------------------------------        
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale) 

        ######===========Construct the grid and update=====================================        
        for i in range(0,self.Size_Vertex_matrix+2): #60 vertical lines    
            x = self.X_off + round(self.X_step_size*i)
            self.cv_img = cv2.line(self.cv_img, (round(x-width/2), y_start), (round(x-width/2 + self.height*math.atan(self.X_tilt*(-1)*3.14/180)), self.height), (255,255,0), 1)

        for i in range(0,self.Size_Vertex_matrix+2): #horizontal lines    
            y = self.Y_off + round(self.Y_step_size*i)
            self.cv_img = cv2.line(self.cv_img, (x_start, round(y-length/2)), (self.width, round(y-length/2 + self.width*math.atan(self.Y_tilt*3.14/180))), (255,255,0), 1)
        
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
#        grid_image = self.cv_img
#        scipy.misc.imsave( folder_path +'/Grid_' + folder_name +'.png', self.cv_img)
        #self.cv_img.save( folder_name +'/Grid_' + folder_name +'.png')
        ####==== CREATE COORDINATE MATRIX======================================
        x=0
        x0=0
        y0=0
        y=0
        i=0
        j=0
        self.Cor_matrix =np.zeros((self.Size_Vertex_matrix+2,self.Size_Vertex_matrix+2,2)) 
        for i in range(0,self.Size_Vertex_matrix+2): #vertical lines x-coordinate at each vertex bottom left corner
            x0 = self.X_off + round(self.X_step_size*i)
            for j in range(0,self.Size_Vertex_matrix+2):        
                y0 = self.Y_off + round(self.Y_step_size*j)        
                x = x0 + round(y0*math.atan(self.X_tilt*(-1)*3.14/180))
                y = y0 + round(x0*math.atan(self.Y_tilt*3.14/180))
                self.Cor_matrix[j,i,0] = x
                self.Cor_matrix[j,i,1] = y             
                
        ####========DISPLAY CROPPED MOMENTS====================================
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale)         
        MFM_img = self.cv_img # cv2.imread(image_path)

        s1 = cv2.resize(cv2.imread(template_path + '/' +'s1.png'), (0,0), fx=image_scale, fy=image_scale)         
        (h1, w1) = s1.shape[:2]
        center = (w1/2, h1/2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, 90, scale)         # Perform the counter clockwise rotation holding at the center
        s4 = cv2.warpAffine(s1, M, (h1, w1)) 
        M = cv2.getRotationMatrix2D(center, 180, scale)         # Perform the counter clockwise rotation holding at the center
        s3 = cv2.warpAffine(s1, M, (h1, w1)) 
        M = cv2.getRotationMatrix2D(center, 270, scale)         # Perform the counter clockwise rotation holding at the center
        s2 = cv2.warpAffine(s1, M, (h1, w1)) 

        imgX = MFM_img[int(self.Cor_matrix[self.m,self.n,1] - self.mnwidth/2):int(self.Cor_matrix[self.m,self.n,1]+ self.mnwidth/2), int(self.Cor_matrix[self.m,self.n,0]+self.X_shift - self.mnlength/2):int(self.Cor_matrix[self.m,self.n,0]+ self.mnlength/2 + self.X_shift), :]
        imgY = MFM_img[int(self.Cor_matrix[self.m,self.n,1])+self.Y_shift:int(self.Cor_matrix[self.m,self.n,1]) + self.mnlength + self.Y_shift, int(self.Cor_matrix[self.m,self.n,0] - self.mnwidth/2):int(self.Cor_matrix[self.m,self.n,0] + self.mnwidth/2), :]

        self.crop_imgX = imgX         
        self.canvasX = tkinter.Canvas(self.frame, width = self.crop_imgX.shape[1], height = self.crop_imgX.shape[0]) # Create a canvas that can fit the above image
        self.canvasX.grid(row=61,column=0) #        self.canvas.pack()
        self.photoX = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgX)) # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage       
        self.canvasX.create_image(0, 0, image=self.photoX, anchor=tkinter.NW) # Add a PhotoImage to the Canvas        

        self.crop_imgY = imgY         
        self.canvasY = tkinter.Canvas(self.frame, width = self.crop_imgY.shape[1], height = self.crop_imgY.shape[0]) # Create a canvas that can fit the above image
        self.canvasY.grid(row=61,column=1) #        self.canvas.pack()
        self.photoY = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(imgY)) # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage       
        self.canvasY.create_image(0, 0, image=self.photoY, anchor=tkinter.NW) # Add a PhotoImage to the Canvas        

        self.res_s1 = np.mean(cv2.matchTemplate(imgX,s1,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
        self.res_s2 = np.mean(cv2.matchTemplate(imgX,s2,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
        self.res_s3 = np.mean(cv2.matchTemplate(imgX,s3,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
        self.res_s4 = np.mean(cv2.matchTemplate(imgX,s4,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
        
        print(self.res_s1, self.res_s2, self.res_s3, self.res_s4)
        maxpos = [self.res_s1, self.res_s2, self.res_s3, self.res_s4].index(max([self.res_s1, self.res_s2, self.res_s3, self.res_s4]))        

        if(maxpos == 0):
            self.label_test    = tkinter.Label(self.frame,text='s1')
            self.label_test.grid(row=60,column=0,sticky=tkinter.S)
        if(maxpos == 1):
            self.label_test    = tkinter.Label(self.frame,text='S2')
            self.label_test.grid(row=60,column=0,sticky=tkinter.S)
        if(maxpos == 2):
            self.label_test    = tkinter.Label(self.frame,text='S3')
            self.label_test.grid(row=60,column=0,sticky=tkinter.S)
        if(maxpos == 3):
            self.label_test    = tkinter.Label(self.frame,text='S4')
            self.label_test.grid(row=60,column=0,sticky=tkinter.S)



######--------------SAVE THE GRID PARAMETERS===================================
    def savegrid(self):
#        scipy.misc.imsave( folder_path +'/Grid_' + folder_name +'.png', grid_image)        
        self.Size_Vertex_matrix = int((self.entry_Size.get()))
        self.X_step_size = float((self.entry_X_step.get()))
        self.Y_step_size = float((self.entry_Y_step.get()))
        self.X_tilt = float((self.entry_X_tilt.get()))
        self.Y_tilt = float((self.entry_Y_tilt.get()))
        self.X_off = int((self.entry_X_off.get()))
        self.Y_off = int((self.entry_Y_off.get()))
        self.X_shift = int((self.entry_X_shift.get()))
        self.Y_shift = int((self.entry_Y_shift.get()))

        file = open( folder_path +'/'+ 'MFM_grid_spec_' + folder_name + '.txt','w') 
        file.write('Size_Vertex_matrix = '+ str(self.Size_Vertex_matrix) + '\n') 
        file.write('X_step_(vertical lines) = '+ str(self.X_step_size) + '\n') 
        file.write('Y_step_(horizontal lines) = '+ str(self.Y_step_size) + '\n') 
        file.write('X_tilt_(vertical lines) = '+ str(self.X_tilt) + '\n') 
        file.write('Y_tilt_(horizontal lines) = '+ str(self.Y_tilt) + '\n')  
        file.write('X_offset_(vertical lines) = '+ str(self.X_off) + '\n') 
        file.write('Y_offset_(horizontal lines) = '+ str(self.Y_off) + '\n')
        file.write('X_shift_(location of X magnet wrt vertex center) = '+ str(self.X_shift) + '\n') 
        file.write('Y_shift_(location of Y magnet wrt vertex center) = '+ str(self.Y_shift) + '\n')
        file.write('Moment_length = '+ str(self.mnlength) + '\n')
        file.write('Moment_width = '+ str(self.mnwidth) + '\n')
        file.write('Bounding_square = '+ str(self.Square) + '\n')
        file.write('Row = '+ str(0) + '\n')
        file.write('Column = '+ str(0) + '\n')
        file.write('Pin_start_Row = '+ str(self.PinStartRow) + '\n')
        file.write('Pin_start_Col = '+ str(self.PinStartCol) + '\n')
        file.write('Pin_Period_X = '+ str(self.PinPeriodX) + '\n')
        file.write('Pin_Period_Y = '+ str(self.PinPeriodY) + '\n')
        file.close()

    def save_img_grid(self):
        y_start = 0
        x_start = 0
        ####----------Reload the original image again------------------------------------        
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale) 

        ######===========Construct the grid and update=====================================        
        for i in range(0,self.Size_Vertex_matrix): #60 vertical lines    
            x = self.X_off + round(self.X_step_size*i)
            self.cv_img = cv2.line(self.cv_img, (round(x-width/2), y_start), (round(x-width/2 + self.height*math.atan(self.X_tilt*(-1)*3.14/180)), self.height), (255,255,0), 1)

        for i in range(0,self.Size_Vertex_matrix): #horizontal lines    
            y = self.Y_off + round(self.Y_step_size*i)
            self.cv_img = cv2.line(self.cv_img, (x_start, round(y-length/2)), (self.width, round(y-length/2 + self.width*math.atan(self.Y_tilt*3.14/180))), (255,255,0), 1)

        scipy.misc.imsave( folder_path +'/Grid_' + folder_name +'.png', self.cv_img)

    def count(self):
        self.color_T1 = int((self.colorT1.get()))
        self.color_T2 = int((self.colorT2.get()))
        self.color_T3 = int((self.colorT3.get()))
        self.color_T4 = int((self.colorT4.get()))
        self.color_P = int((self.colorPin.get()))
        self.color_RP = int((self.colorRevPin.get()))
        self.spin_direction_index = self.spin_direction.get()
        print(self.spin_direction_index)
        
        s1 = cv2.resize(cv2.imread(template_path + '/' + 's1.png'), (0,0), fx=image_scale, fy=image_scale)         
        (h1, w1) = s1.shape[:2]
        center = (w1/2, h1/2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, 90, scale)         # Perform the counter clockwise rotation holding at the center
        s4 = cv2.warpAffine(s1, M, (h1, w1)) 
        M = cv2.getRotationMatrix2D(center, 180, scale)         # Perform the counter clockwise rotation holding at the center
        s3 = cv2.warpAffine(s1, M, (h1, w1)) 
        M = cv2.getRotationMatrix2D(center, 270, scale)         # Perform the counter clockwise rotation holding at the center
        s2 = cv2.warpAffine(s1, M, (h1, w1)) 
        
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale)         

        MFM_img = self.cv_img
#        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        self.Mx_matrix = np.zeros((self.Size_Vertex_matrix,self.Size_Vertex_matrix))
#        self.My_matrix = np.zeros((self.Size_Vertex_matrix,self.Size_Vertex_matrix))
        lattice_M = self.Size_Vertex_matrix
        lattice_N = self.Size_Vertex_matrix
        mm=0
        nn=0
        for nn in range(0,self.Size_Vertex_matrix):    
            for mm in range(0,self.Size_Vertex_matrix):       
                self.imgX = MFM_img[int(self.Cor_matrix[mm,nn,1] - self.mnwidth/2):int(self.Cor_matrix[mm,nn,1]+ self.mnwidth/2), int(self.Cor_matrix[mm,nn,0]+self.X_shift - self.mnlength/2):int(self.Cor_matrix[mm,nn,0]+ self.mnlength/2 + self.X_shift), :]
#                self.imgX = MFM_img[int(self.Cor_matrix[mm,nn,1] - self.mnwidth/2):int(self.Cor_matrix[mm,nn,1]+ self.mnwidth/2), int(self.Cor_matrix[mm,nn,0])+self.X_shift:int(self.Cor_matrix[mm,nn,0]) + self.mnlength + self.X_shift, :]
#                self.imgY = MFM_img[int(self.Cor_matrix[mm,nn,1])+self.Y_shift:int(self.Cor_matrix[mm,nn,1]) + self.mnlength + self.Y_shift, int(self.Cor_matrix[mm,nn,0] - self.mnwidth/2):int(self.Cor_matrix[mm,nn,0] + self.mnwidth/2), :]
                self.res_s1 = np.mean(cv2.matchTemplate(self.imgX,s1,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
                self.res_s2 = np.mean(cv2.matchTemplate(self.imgX,s2,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
                self.res_s3 = np.mean(cv2.matchTemplate(self.imgX,s3,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
                self.res_s4 = np.mean(cv2.matchTemplate(self.imgX,s4,cv2.TM_CCOEFF_NORMED))#cv2.TM_CCOEFF_NORMED)    
                
                maxpos = [self.res_s1, self.res_s2, self.res_s3, self.res_s4].index(max([self.res_s1, self.res_s2, self.res_s3, self.res_s4]))        
                if(maxpos == 0):
                    self.Mx_matrix[mm,nn] = 1
                if(maxpos == 1):
                    self.Mx_matrix[mm,nn] = 2
                if(maxpos == 2):
                    self.Mx_matrix[mm,nn] = 3
                if(maxpos == 3):
                    self.Mx_matrix[mm,nn] = 4

        np.savetxt(folder_path + '/'+'S_mfm.txt', self.Mx_matrix, fmt='%1i')

        Potts_domain = PottsMappinDI(self.Mx_matrix, self.spin_direction_index)
        Potts_domain.save(folder_path + '/'+ folder_name+'_Potts.png')

        self.photo = PIL.ImageTk.PhotoImage(image = Potts_domain)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

#                                
#        n0Sx = self.Mx_matrix
#        n0Sy = self.My_matrix
#        Sx = np.zeros((2*(n0Sx.shape[0])+1,2*(n0Sx.shape[1])+1))
#        Sx[1::2,1::2] = n0Sx
#        Sx.astype(int)
#        Sx = np.delete(Sx, (0), axis=0)
#        Sx = np.delete(Sx, (-1), axis=1)
#        
#        Sy = np.zeros((2*(n0Sx.shape[0])+1,2*(n0Sx.shape[1])+1))
#        Sy[1::2,1::2] = n0Sy
#        Sy.astype(int)
#        Sy = np.delete(Sy, (-1), axis=0)
#        Sy = np.delete(Sy, (0), axis=1)
#        
#        Wholearray = np.stack((Sx,Sy))
#        Sxy = Wholearray[:,0:Sx.shape[0]-1,0:Sx.shape[0]-1]
#        S = Sxy[0,:,:] + Sxy[1,:,:] 
#        Sx = Sxy[0,:,:]
#        Sy = Sxy[1,:,:]

#        FinalDomainState = DomainImage(Sxy[0,:,:], Sxy[1,:,:])
#        FinalDomainState.save(folder_path + '/'+ folder_name+'_digitized.png')
#        FinalDomainState.show()

        
#        self.V = np.zeros((Sxy.shape[1],Sxy.shape[2]))
#        for xi in range (0,Sxy.shape[1]-1):
#            for yi in range (0,Sxy.shape[2]-1):
#                if (Sxy[0,yi,xi] != 0): # x compoment
#                    if (0<xi<Sxy.shape[1]-1):
#                        vert_right = xi+1
#                        vert_left  = xi-1
#                        self.V[yi,xi+1] = sort_vertex_btype(yi,vert_right,Sxy,Sxy.shape[1],Sxy.shape[2])                
#                        self.V[yi,xi-1] = sort_vertex_btype(yi,vert_left,Sxy,Sxy.shape[1],Sxy.shape[2])
#        
#                if (Sxy[1,yi,xi] != 0):
#                    if (0<yi<Sxy.shape[2]-1):
#                        vert_top = yi-1
#                        vert_bot = yi+1
#                        self.V[yi+1,xi] = sort_vertex_btype(vert_bot,xi,Sxy,Sxy.shape[1],Sxy.shape[2])
#                        self.V[yi-1,xi] = sort_vertex_btype(vert_top,xi,Sxy,Sxy.shape[1],Sxy.shape[2]) 
        
                
#        VT = [10,11,20,21,22,23,30,31,32,33,34,35,36,37,40,41]
#        Total_vertices = 0                
        file = open( folder_path +'/'+ 'Vertex_statistics_' + folder_name + '.txt','w') 
        #file.write('Threshold = '+ str(threshold) + '\n') 
        file.write('Effective_vertex_matrix_size  = ' + str(lattice_M-2) +'x'+str(lattice_N-2) + '\n') 
        file.write('Xcell  = ' +str(lattice_N-1) + '\n') 
        file.write('Ycell  = ' +str(lattice_N-1) + '\n') 
        
#        Xcell = self.Size_Vertex_matrix+1
#        Ycell = self.Size_Vertex_matrix+1
#        Vcounts =[]
#        Vcount_std = []
#        for i in range(0,16):
#            file.write('T'+str(VT[i]) +' = '+ str(np.count_nonzero(self.V.flatten() == VT[i])) + '\n') 
#            print('T'+str(VT[i]) +' = '+ str(np.count_nonzero(self.V.flatten() == VT[i])))
#            Vcounts.append(np.count_nonzero(self.V.flatten() == VT[i]))
#            Total_vertices = Total_vertices + np.count_nonzero(self.V.flatten() == VT[i])
#            
#            V1 = self.V[0:round(self.V.shape[0]/2), 0:round(self.V.shape[1]/2)]
#            V2 = self.V[round(self.V.shape[0]/2):self.V.shape[0]-1, 0:round(self.V.shape[1]/2)]
#            V3 = self.V[0:round(self.V.shape[0]/2), round(self.V.shape[1]/2):self.V.shape[1]-1]
#            V4 = self.V[round(self.V.shape[0]/2):self.V.shape[0]-1, round(self.V.shape[1]/2):self.V.shape[1]-1]
            
#            n1 = np.count_nonzero(V1.flatten() == VT[i])
#            n2 = np.count_nonzero(V2.flatten() == VT[i])
#            n3 = np.count_nonzero(V3.flatten() == VT[i])
#            n4 = np.count_nonzero(V4.flatten() == VT[i])            
#            Vcount_std.append(np.std([n1,n2,n3,n4]))

        ###====Error estimate in magnetic moment===============================        
#        mx1 = np.mean(n0Sx[0:round(n0Sx.shape[0]/2), 0:round(n0Sx.shape[1]/2)])
#        mx2 = np.mean(n0Sx[0:round(n0Sx.shape[0]/2), round(n0Sx.shape[1]/2):n0Sx.shape[1]])
#        mx3 = np.mean(n0Sx[round(n0Sx.shape[0]/2):n0Sx.shape[0], 0:round(n0Sx.shape[1]/2)])
#        mx4 = np.mean(n0Sx[round(n0Sx.shape[0]/2):n0Sx.shape[0], round(n0Sx.shape[1]/2):n0Sx.shape[1]])
#        err_mx = np.std([mx1,mx2,mx3,mx4])

#        my1 = np.mean(n0Sy[0:round(n0Sy.shape[0]/2), 0:round(n0Sy.shape[1]/2)])
#        my2 = np.mean(n0Sy[0:round(n0Sy.shape[0]/2), round(n0Sy.shape[1]/2):n0Sy.shape[1]])
#        my3 = np.mean(n0Sy[round(n0Sy.shape[0]/2):n0Sy.shape[0], 0:round(n0Sy.shape[1]/2)])
#        my4 = np.mean(n0Sy[round(n0Sy.shape[0]/2):n0Sy.shape[0], round(n0Sy.shape[1]/2):n0Sy.shape[1]])
#        err_my = np.std([my1,my2,my3,my4])
#            
#        print('Avg mx = ', np.mean(n0Sx))
#        print('Err mx = ',  err_mx)
#        print('Avg my = ', np.mean(n0Sy))
#        print('Err my = ', err_my)           
#        print('Total vertices = '+ str(Total_vertices))
#
#        file.write('Total_T1 = ' + str(sum(Vcounts[0:2])) + '\n')
#        file.write('Total_T2 = ' + str(sum(Vcounts[2:6])) + '\n')
#        file.write('Total_T3 = ' + str(sum(Vcounts[6:14])) + '\n')
#        file.write('Total_T4 = ' + str(sum(Vcounts[14:16])) + '\n')
#
#        n1 = np.count_nonzero(V1.flatten() == VT[0]) + np.count_nonzero(V1.flatten() == VT[1])
#        n2 = np.count_nonzero(V2.flatten() == VT[0]) + np.count_nonzero(V2.flatten() == VT[1])
#        n3 = np.count_nonzero(V3.flatten() == VT[0]) + np.count_nonzero(V3.flatten() == VT[1])
#        n4 = np.count_nonzero(V4.flatten() == VT[0]) + np.count_nonzero(V4.flatten() == VT[1])            
#        Vcount_std.append(np.std([n1,n2,n3,n4]))
#        file.write('%_T1 = ' + str(100*sum(Vcounts[0:2])/Total_vertices) + '\n')
#        file.write('Err_%_T1 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) + '\n')
#        print('%_T1 = ' + str(100*sum(Vcounts[0:2])/Total_vertices) )
#        print('Err_%_T1 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) )

#        n1 = np.count_nonzero(V1.flatten() == VT[2]) + np.count_nonzero(V1.flatten() == VT[3]) + np.count_nonzero(V1.flatten() == VT[4]) + np.count_nonzero(V1.flatten() == VT[5])
#        n2 = np.count_nonzero(V2.flatten() == VT[2]) + np.count_nonzero(V2.flatten() == VT[3]) + np.count_nonzero(V2.flatten() == VT[4]) + np.count_nonzero(V2.flatten() == VT[5])
#        n3 = np.count_nonzero(V3.flatten() == VT[2]) + np.count_nonzero(V3.flatten() == VT[3]) + np.count_nonzero(V3.flatten() == VT[4]) + np.count_nonzero(V3.flatten() == VT[5])
#        n4 = np.count_nonzero(V4.flatten() == VT[2]) + np.count_nonzero(V4.flatten() == VT[3]) + np.count_nonzero(V4.flatten() == VT[4]) + np.count_nonzero(V4.flatten() == VT[5])           
#        Vcount_std.append(np.std([n1,n2,n3,n4]))
#        file.write('%_T2 = ' + str(100*sum(Vcounts[2:6])/Total_vertices) + '\n')
#        file.write('Err_%_T2 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) + '\n')
#        print('%_T2 = ' + str(100*sum(Vcounts[2:6])/Total_vertices) )
#        print('Err_%_T2 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) )
#
#        n1 = np.count_nonzero(V1.flatten() == VT[6]) + np.count_nonzero(V1.flatten() == VT[7]) + np.count_nonzero(V1.flatten() == VT[8]) + np.count_nonzero(V1.flatten() == VT[9])+\
#             np.count_nonzero(V1.flatten() == VT[10]) + np.count_nonzero(V1.flatten() == VT[11]) + np.count_nonzero(V1.flatten() == VT[12]) + np.count_nonzero(V1.flatten() == VT[13])
#        n2 = np.count_nonzero(V2.flatten() == VT[6]) + np.count_nonzero(V2.flatten() == VT[7]) + np.count_nonzero(V2.flatten() == VT[8]) + np.count_nonzero(V2.flatten() == VT[9])+\
#             np.count_nonzero(V2.flatten() == VT[10]) + np.count_nonzero(V2.flatten() == VT[11]) + np.count_nonzero(V2.flatten() == VT[12]) + np.count_nonzero(V2.flatten() == VT[13])
#        n3 = np.count_nonzero(V3.flatten() == VT[6]) + np.count_nonzero(V3.flatten() == VT[7]) + np.count_nonzero(V3.flatten() == VT[8]) + np.count_nonzero(V3.flatten() == VT[9])+\
#             np.count_nonzero(V3.flatten() == VT[10]) + np.count_nonzero(V3.flatten() == VT[11]) + np.count_nonzero(V3.flatten() == VT[12]) + np.count_nonzero(V3.flatten() == VT[13])
#        n4 = np.count_nonzero(V4.flatten() == VT[6]) + np.count_nonzero(V4.flatten() == VT[7]) + np.count_nonzero(V4.flatten() == VT[8]) + np.count_nonzero(V4.flatten() == VT[9])+\
#             np.count_nonzero(V4.flatten() == VT[10]) + np.count_nonzero(V4.flatten() == VT[11]) + np.count_nonzero(V4.flatten() == VT[12]) + np.count_nonzero(V4.flatten() == VT[13])
#        Vcount_std.append(np.std([n1,n2,n3,n4]))
#        file.write('%_T3 = ' + str(100*sum(Vcounts[6:14])/Total_vertices) + '\n')
#        file.write('Err_%_T3 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) + '\n')
#        print('%_T3 = ' + str(100*sum(Vcounts[6:14])/Total_vertices) )
#        print('Err_%_T3 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) )
#
#        n1 = np.count_nonzero(V1.flatten() == VT[14]) + np.count_nonzero(V1.flatten() == VT[15])
#        n2 = np.count_nonzero(V2.flatten() == VT[14]) + np.count_nonzero(V2.flatten() == VT[15])
#        n3 = np.count_nonzero(V3.flatten() == VT[14]) + np.count_nonzero(V3.flatten() == VT[15])
#        n4 = np.count_nonzero(V4.flatten() == VT[14]) + np.count_nonzero(V4.flatten() == VT[15])            
#        Vcount_std.append(np.std([n1,n2,n3,n4]))
#        file.write('%_T4 = ' + str(100*sum(Vcounts[15:16])/Total_vertices) + '\n')
#        file.write('Err_%_T4 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices) + '\n')
#        print('%_T4 = ' + str(100*sum(Vcounts[15:16])/Total_vertices) )
#        print('Err_%_T4 = ' + str(100*sum(Vcount_std[0:4])/Total_vertices))

        
#        file.write('Percentage_T1 = ' + str(100*sum(Vcounts[0:2])/Total_vertices) + '\n')
#        file.write('Percentage_T2 = ' + str(100*sum(Vcounts[2:6])/Total_vertices) + '\n')
#        file.write('Percentage_T3 = ' + str(100*sum(Vcounts[6:14])/Total_vertices) + '\n')
#        file.write('Percentage_T4 = ' + str(100*sum(Vcounts[14:16])/Total_vertices) + '\n')
#        file.write('Total_vertices = ' + str(Total_vertices) + '\n')
        
#        file.write('Avg_mx = ' + str(np.mean(n0Sx)) + '\n')
#        file.write('Err_mx = ' + str(err_mx) + '\n')
#        file.write('Avg_my = ' + str(np.mean(n0Sy)) + '\n')
#        file.write('Err_my = ' + str(err_my) + '\n')

#        for i in range(0,16):
#            file.write('%_T'+str(VT[i]) +' = '+ str(100*np.count_nonzero(self.V.flatten() == VT[i])/Total_vertices) + '\n') 
#            print('%_T'+str(VT[i]) +' = '+ str(100*np.count_nonzero(self.V.flatten() == VT[i])/Total_vertices))
#
#            n1 = np.count_nonzero(V1.flatten() == VT[i])
#            n2 = np.count_nonzero(V2.flatten() == VT[i])
#            n3 = np.count_nonzero(V3.flatten() == VT[i])
#            n4 = np.count_nonzero(V4.flatten() == VT[i])            
#            Vcount_std.append(np.std([n1,n2,n3,n4]))
#            print('Err_%_T'+str(VT[i]) +' = '+ str(100*Vcount_std[i]/Total_vertices))
#            file.write('Err_%_T'+str(VT[i]) +' = '+ str(100*Vcount_std[i]/Total_vertices) + '\n') 
       
        
#        self.Sx = Sx
#        self.Sy = Sy
#        self.Sxy = Sxy
#        self.S = Sx+Sy
#        np.savetxt(folder_path + '/'+'Sx_mfm.txt', Sxy[0,:,:], fmt='%1i')
#        np.savetxt(folder_path + '/'+'Sy_mfm.txt', Sxy[1,:,:], fmt='%1i')
#        np.savetxt(folder_path + '/'+'V_mfm.txt', self.V, fmt='%1i')
#        
 
        #####===========Highlight vertex types=====================================        
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = cv2.resize(self.cv_img, (0,0), fx=image_scale, fy=image_scale) 
#        n0V = self.V[~(self.V==0).all(1)]
#        n0V = np.transpose((np.transpose(n0V)[~np.all(np.transpose(n0V) == 0, axis=1)])) 
#self.Mx_matrix
        for j in range(0,self.Size_Vertex_matrix): #60 vertical lines
            for i in range(0,self.Size_Vertex_matrix): #horizontal lines
#                if(i>0 and i<self.Size_Vertex_matrix+1 and j>0 and j<self.Size_Vertex_matrix+1):
                if(self.color_T1 ==1):                    
                    if (self.Mx_matrix[j,i] == 1):
                        self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (255,0,0), 1)
                if(self.color_T2 ==1):                    
                    if (self.Mx_matrix[j,i] == 2):
                        self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (0,0,255), 1)
                if(self.color_T3 ==1):                    
                    if (self.Mx_matrix[j,i] == 3):
                        self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (0,255,0), 1)
                if(self.color_T4 ==1):                    
                    if (self.Mx_matrix[j,i] == 4):
                        self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (255,255,0), 1)
                    
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        scipy.misc.imsave( folder_path +'/PottMFM_' + folder_name +'.png', self.cv_img)
        

#        n0V1, n0V2 = GroundState(2*self.Size_Vertex_matrix+4, 2*self.Size_Vertex_matrix+4)
#        print(n0V1.shape)
#        print(n0V.shape)
#        for j in range(0,self.Size_Vertex_matrix+2): #60 vertical lines
#            for i in range(0,self.Size_Vertex_matrix+2): #horizontal lines
#                if(i>0 and i<self.Size_Vertex_matrix+1 and j>0 and j<self.Size_Vertex_matrix+1):
#                    if(self.color_T1 ==1):
#                        if (n0V[j-1,i-1] == 10 or n0V[j-1,i-1] == 11):
#                            if(n0V[j-1,i-1]==n0V1[j-1,i-1]):
#                                self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (255,0,0), 1)
#                            else:
#                                self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (155,0,0), 1)                                
#                    if(self.color_T2 ==1):                    
#                        if (n0V[j-1,i-1]==20 or n0V[j-1,i-1]==21 or n0V[j-1,i-1]==22 or n0V[j-1,i-1]==23):
#                            self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (0,0,255), 1)
#                    if(self.color_T3 ==1):                    
#                        if (n0V[j-1,i-1]==30 or n0V[j-1,i-1]==31 or n0V[j-1,i-1]==32 or n0V[j-1,i-1]==33 or n0V[j-1,i-1]==34 or n0V[j-1,i-1]==35 or n0V[j-1,i-1]==36 or n0V[j-1,i-1]==37):
#                            self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (0,255,0), 1)
#                    if(self.color_T4 ==1):                    
#                        if (n0V[j-1,i-1]==40 or n0V[j-1,i-1]==41):
#                            self.cv_img = cv2.rectangle(self.cv_img, (int(self.Cor_matrix[j,i,0]-self.Square/2), int(self.Cor_matrix[j,i,1]-self.Square/2)), (int(self.Cor_matrix[j,i,0]+self.Square/2),int(self.Cor_matrix[j,i,1]+self.Square/2)), (255,255,0), 1)
#                    
#        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
#        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
#        scipy.misc.imsave( folder_path +'/T1_domains_' + folder_name +'.png', self.cv_img)
       
        file.close() 

#    def result(self):    
#        return self.Size_Vertex_matrix, self.X_step_size, self.Y_step_size, self.X_tilt, self.Y_tilt, self.X_off, self.Y_off, self.X_shift, self.Y_shift, self.mnwidth, self.mnlength, self.Cor_matrix, self.Mx_matrix, self.My_matrix,self.height, self.width, self.V, self.Sx, self.Sy, self.Sxy,self.S
        
##======= Create a window and pass it to the Application object================
Mat = App(tkinter.Tk(), "Tkinter and OpenCV")


#Size_Vertex_matrix = Mat.result()[0]
#X_step_size        = Mat.result()[1] # vertical lines 
#Y_step_size        = Mat.result()[2] #37.4 # horizontal lines
#X_tilt             = Mat.result()[3] # tilt vertical lines +ve countercloswise
#Y_tilt             = Mat.result()[4] # tilt horizontal line -ve counter clock
#X_off              = Mat.result()[5] #itial position of vertical grid line
#Y_off              = Mat.result()[6] #intiial position of horizontal grid line
#X_shift            = Mat.result()[7] # position of magnetic element from vertex center
#Y_shift            = Mat.result()[8] # position of magnetic element from vertex center
#wd                 = Mat.result()[9]
#ln                 = Mat.result()[10]
#Cor_matrix         = Mat.result()[11]
#
#Mx = Mat.result()[12]
#My = Mat.result()[13]
#ImageHeight = Mat.result()[14]
#ImageWidth = Mat.result()[15]

#Vb = Mat.result()[16]
#Sx = Mat.result()[17]
#Sy = Mat.result()[18]
#Sxy = Mat.result()[19]
#S = Mat.result()[20]