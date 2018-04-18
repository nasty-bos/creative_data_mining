import sqlite3
import glob
import io
import requests
import pandas as pd
import csv
import sys

if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

url = 'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/03ec9d0a-b16f-4e78-8e4f-2da4970efbb6/download/fahrzeiten_soll_ist_20180325_20180331.csv'
s = requests.get(url).content
c = pd.read_csv(io.StringIO(s.decode('utf-8')))
print(c)

# with urlopen(url) as url:
#     s = url.read()
# # print(s)
#
# df = pd.read_csv(s)
# # q = df.shape[1] - 2
# # c = df.shape[0]
# print(df)
# # print(q)
# print(c)


# conn = sqlite3.connect('delay.db')
# ss.to_sql(delay, conn, if_exists='append', index=False)

# cur = con.cursor()
# cur.execute("CREATE TABLE t (s);")  # use your column names here
#
# with open('s', 'rb') as fin:  # `with` statement available in 2.5+
#     # csv.DictReader uses first line in file for column headings by default
#     dr = csv.DictReader(fin)  # comma is default delimiter
#     to_db = [(i[''], i['col2']) for i in dr]
#
# cur.executemany("INSERT INTO t (col1, col2) VALUES (?, ?);", to_db)
# con.commit()
# con.close()
# db.close()
