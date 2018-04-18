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


conn = sqlite3.connect('c.db')
cur = conn.cursor()
# ss.to_sql(delay, conn, if_exists='append', index=False)

cmd = "CREATE TABLE IF NOT EXISTS %s (=)"

cur.execute("select * from linie")  # use your column names here
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
