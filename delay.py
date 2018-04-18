import requests
import pandas as pd


def mask(df, key, value):
    return df[df[key] == value]


url = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/03ec9d0a-b16f-4e78-8e4f-2da4970efbb6/download/fahrzeiten_soll_ist_20180325_20180331.csv'
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c.head())

df = pd.DataFrame(data=c)
df.drop(df.columns[[range(3, 10), range(13, 33)]], axis=1, inplace=True)
print(df.head())

pd.DataFrame.mask = mask
df.mask(df, 'linie', 69)
