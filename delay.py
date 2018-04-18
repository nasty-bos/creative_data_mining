import io
import requests
import pandas as pd
import numpy as np


def mask(df, key, value):
    return df[df[key] == value]


url = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/03ec9d0a-b16f-4e78-8e4f-2da4970efbb6/download/fahrzeiten_soll_ist_20180325_20180331.csv'
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c.head())

df = pd.DataFrame(data=c)
df.drop(df.columns[
            [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
             33]], axis=1, inplace=True)
print(df.head())

pd.DataFrame.mask = mask
df = df.mask('linie', 69)
check = pd.DataFrame(data=df)
col = check.shape[1]
row = check.shape[0]
print(row)
print(col)
# dp['ist_an_von']
# df['ist_an_von'] = pd.to_datetime(df["ist_an_von"], unit='s')
print(df.head())

df.loc[:, 'diff'] = (df['ist_an_von'] - df['soll_an_von'])
print(df.head())

dp = df.copy()
dp.reset_index(drop=True, inplace=True)
print(dp.head())

# time: not efficient at all to calculate the time like that..
for i in range(1,row):
    m, s = divmod(dp.loc[i,'ist_an_von'], 60)
    h, m = divmod(m, 60)
    dp.loc[i,'time'] = h
    print(dp.head())
#     dp[i,'time'] = str(datetime.timedelta(seconds=dp[i,'ist_an_von']))
# dp.loc[i,'time']=(dp.loc[i,'ist_an_von'])
# dp.drop(dp.columns[[1,3,4]], axis=1, inplace=True)
# dp.loc[:, 'diff'] = dp['diff']/
# dp.head()