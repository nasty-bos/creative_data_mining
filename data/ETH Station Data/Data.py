#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 20:24:02 2018

@author: Nicol
"""

#%%Weather Data
#creates the file 'Weather data.csv' @ the folder 'Weather'
#columns: Time [min]    Air temperature [deg C]    Rain accumulated during 10 min measurement interval [mm]

import pandas as pd
import csv
import os
import glob
import shutil

f=sorted(glob.glob('IAC-Met-HBerg_2018-*.dat'))           
os.makedirs(r'./Weather')                         
newpath=r'./Weather' 

final=pd.DataFrame({'' : []})

for i in f:
    
    with open(i,'r') as input_file:
        lines = input_file.readlines()
        newLines = []
        for line in lines:
            newLine = line.strip().split()
            newLines.append(newLine)

        with open('output.csv', 'w') as output_file:
            file_writer = csv.writer(output_file)
            file_writer.writerows(newLines)

    import pdb; pdb.set_trace()
    meta = datetime.datetime.strptime(pd.read_csv('output.csv',header=None, skiprows=range(6+1), usecols=[0]), "%Y %m %d")        
    data=pd.read_csv('output.csv',header=None, skiprows=43, usecols=[0,2,7])
    final=pd.concat([final, data])


final.to_csv(newpath + '_Data.csv', index = False, header=None)
file=newpath + '_Data.csv' 
shutil.move(file,newpath)  
os.remove('output.csv')   
