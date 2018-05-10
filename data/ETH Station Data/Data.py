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
import datetime

f=sorted(glob.glob('IAC-Met-HBerg_2018-*.dat'))           

if not os.path.exists(os.path.join('Weather')):
    os.makedirs(r'./Weather')                         

path = r'./Weather' 

final=pd.DataFrame({'' : []})

container = []
for i in f:

    with open(i,'r') as input_file:
        ls = i.split('_')
        date = datetime.datetime.strptime(ls[1].replace('.dat', ''), '%Y-%m-%d')

        print("Processing date %s" %ls[1])

        lines = input_file.readlines()
        newLines = []
        for line in lines:
            newLine = line.strip().split()
            newLines.append(newLine)

        cols = newLines[41]
        data = newLines[45: ]

        df = pd.DataFrame.from_records(data)
        df.columns = cols

        hours = df.time.astype(int) // 60
        minutes = df.time.astype(int).mod(60) 
        dateTime = [date.replace(minute=mn, hour=hr) for (hr, mn) in pd.concat([hours, minutes], axis=1).values]
        df.index = dateTime
        container.append(df)

        del df, data, cols, hours, minutes, dateTime

final = pd.concat(container, axis=0)
final.to_csv(os.path.join(path, 'dated_weather_data.csv'))
