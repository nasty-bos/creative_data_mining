import os
import datetime
import pandas 

##################################################################################
def data_dir():
	return "./data/"


##################################################################################
def download_delay_data():
	## Import of delay data
	listpaths = [
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/a265b5d8-287f-4d22-88b2-f3a1770e1a4a/download/fahrzeiten_soll_ist_20180225_20180303.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/03ec9d0a-b16f-4e78-8e4f-2da4970efbb6/download/fahrzeiten_soll_ist_20180325_20180331.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/c88a3801-c6fc-4d32-8ece-e269899be497/download/fahrzeiten_soll_ist_20180318_20180324.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/eb403fd1-8f8b-475e-98aa-f04ee3b255ba/download/fahrzeiten_soll_ist_20180311_20180317.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/1ac13127-fcde-4ac2-8462-50f348fd28fe/download/fahrzeiten_soll_ist_20180218_20180224.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/97e59d2a-83ec-438f-ae6f-0fe85d9bc1e6/download/fahrzeiten_soll_ist_20180304_20180310.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/b45b383e-4b0d-4ad0-8bee-e958c5e7360a/download/fahrzeiten_soll_ist_20180121_20180127.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/f17a950d-5be5-4b00-bafd-3c859afcc6cc/download/fahrzeiten_soll_ist_20180204_20180210.csv',
	    'https://data.stadt-zuerich.ch/dataset/vbz_fahrzeiten_ogd/resource/a38c5d0f-b732-4f5a-9786-eb01a2ffa0bb/download/fahrzeiten_soll_ist_20180211_20180217.csv']
	frame = pandas.DataFrame()
	list_ = []

	for path_ in listpaths:
		print('downloading delay data - %s' %path_.split('/')[-1])
		df = pandas.read_csv(path_, index_col=None)
		df.to_csv(os.path.join(data_dir(), path_.split('/')[-1]))
		list_.append(df)

	return frame


##################################################################################
def get_delay_data():
	files = os.listdir(os.path.join(data_dir()))
	delayFiles = [i for i in files if i.startswith('fahrzeiten')]

	store = list()
	for csv in delayFiles:
		result = pandas.read_csv(
					os.path.join(data_dir(), csv), 
					header=0, 
					index_col=0, 
					parse_dates=['betriebsdatum', 'datum_von'],
					date_parser=lambda x: datetime.datetime.strptime(x, '%d.%m.%y'))
		store.append(result)

	return pandas.concat(store, axis=0)


##################################################################################
def get_lineie_69_data():
	fullPath = os.path.join(data_dir(), 'linie_bus_69.csv')
	return pandas.read_csv(
				fullPath, 
				header=0, 
				index_col=0,
				parse_dates=['betriebsdatum', 'datum_von'],
				date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))


##################################################################################
def get_weather_data():
	fullPath = os.path.join(data_dir(), 'agrometeo-data.csv')
	result = pandas.read_csv(os.path.join(
				data_dir(), 'agrometeo-data.csv'), 
				names=['datetime', 'temp_degrees_c_mittel', 'niederschlag_mm'], 
				sep=';', header=None, 
				skiprows=[0, 1, 2],
				index_col=0,
				parse_dates=True,
				date_parser=lambda x: datetime.datetime.strptime(x, '%d.%m.%Y %H:%M'))

	return result


##################################################################################
def fourier_frequency_filter():

	return {'finished': False}
	
