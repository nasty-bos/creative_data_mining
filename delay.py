import io
import shutil
import requests
import pandas as pd
import numpy as np
import csv
import glob
import os
from sklearn import linear_model
import numpy
from datetime import datetime

def mask(df, key, value):
    return df[df[key] == value]

# import data of delays
url = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/03ec9d0a-b16f-4e78-8e4f-2da4970efbb6/download/fahrzeiten_soll_ist_20180325_20180331.csv'
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col=None)

# time difference & clean up
df = pd.DataFrame(data=c)
df.drop(df.columns[
            [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             33]], axis=1, inplace=True)
pd.DataFrame.mask = mask
df = df.mask('linie', 69)
df.loc[:, 'diff'] = (df['ist_an_von'] - df['soll_an_von'])
dp = df.copy()
dp.reset_index(drop=True, inplace=True)
dp.loc[:, 'time'] = dp.loc[:, 'soll_an_von'].copy().astype(float)
dp.loc[:, 'time'] = pd.to_datetime(dp.loc[:, 'time'], unit='s')
dp.loc[:, 'time'] = dp.loc[:, 'time'].dt.strftime('%H:%M')
df1 = dp.copy()
df1.drop(dp.columns[[3, 4]], axis=1, inplace=True)
df1['time'] = pd.to_datetime(df1['betriebsdatum'] + ' ' + df1['time'])
df1.drop(df1.columns[[2]], axis=1, inplace=True)
df1.loc[:, 'diff'] = df1.loc[:, 'diff'].apply(pd.to_numeric, errors='coerce', downcast='float')

# weather import & clean-up
pd.options.mode.chained_assignment = None
# input folder
path = r'./weather/*.csv'
# import csv as dataframe
new_cols = ['weather']
we = pd.read_csv(glob.glob('./weather/*.csv')[0], header=None, names=new_cols)
wet1 = pd.DataFrame(data=we)
# clean-up
wet = wet1.iloc[3:]
wet.loc[:, 'time'] = wet.weather.str.split(';').str.get(0)
wet.loc[:, 'rain'] = wet.weather.str.split(';').str.get(1)
wet.drop(wet.columns[[0]], axis=1, inplace=True)
wet.loc[:, 'time'] = pd.to_datetime(wet.loc[:, 'time'])
wet.loc[:, 'rain'] = wet.loc[:, 'rain'].apply(pd.to_numeric, errors='coerce')

q = wet.shape[1]
o = wet.shape[0]
print('Number of columns: %d' % q)
print('Number of rows: %d' % o)
print(wet.head())
print(wet.dtypes)


# match of delay and weather frames
# def nearest(items, pivot):
#     return min(items, key=lambda x: abs(x - pivot))
#
#
# df1.apply(nearest(wet.loc[:, 'time'], df1.loc[:, 'time']))
# df1.iloc[df1.index.get_loc(wet.loc[:,'time'],method='nearest')]
# df1['time'] = pd.DatetimeIndex(wet['time']).normalize()
# df1.apply(lambda col: col.drop_duplicates().reset_index(drop=True))
# pred = pd.merge(df1, wet).head()
