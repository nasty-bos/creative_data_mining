import os
import pandas
import numpy
import scipy.fftpack
import datetime

import matplotlib.pyplot as plt
import data as dt

##################################################################################
def main():

	# === Read DELAYS and WEATHER data
	delays = dt.get_lineie_69_data()
	weather = dt.get_weather_data()

	# === Focus on BUS 69
	mask = delays.linie == 69
	delays = delays[mask]
	delays.reset_index(drop=True, inplace=True)

	# === Extract exact time delays
	delays.loc[:, 'diff'] = delays.ist_an_von - delays.soll_an_von
	delays.loc[:, 'time'] = pandas.to_datetime(delays.soll_an_von.copy().astype(float), errors='coerce', unit='s')
	delays.time = delays.time.dt.strftime('%H:%M')
	delays.loc[:, 'datetime'] = pandas.to_datetime(delays.datum_von.astype(str) + ' ' + delays.time)
	delays.datetime = delays.datetime.dt.round('60min')

	# === Show delay pattern as a function of time of day
	plt.figure()
	delays.groupby('time').sum()['diff'].plot()

	# === Merge with WEATHER data 
	weatherDelays = delays.merge(weather, left_on='datetime', right_index=True, how='left')
	weatherDelays.to_csv(os.path.join(dt.data_dir(), 'weather_delays_merged.csv'))

	mask = weatherDelays.datetime > datetime.datetime(2018,2,4)
	weatherDelays = weatherDelays[mask]
	del mask
	
	weatherDelaysHourly = weatherDelays.groupby('datetime').sum()

	# === Estimate DAILY SEASONALITY using Fourier transform
	'''
	Description:
		Fourier transform of time-series data in time domain (yt, xt) to frequency domain (yf, xf):

	Arguments:
		:param n: (float) number of data points / observations  
		:param T: (float) maximum frequency of data i.e. 1H, 1m, 1s 	
	'''
	n, m = weatherDelaysHourly.shape
	T = 1/n
	yf = scipy.fftpack.fft(weatherDelays['diff'].values)
	xf = numpy.linspace(0, 1/2.0 * T, n/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/n * numpy.abs(yf[:n//2]))

	# TODO Buiild function that 1. Picks data window, 2. FFT, 3. Removes frequency in xf domain, 4. IFFT, 5. Corr

	# === Try to remove DAILY SEASONALITY by subtracting previous day's value
	timeDelta = datetime.timedelta(hours=24)
	temp = weatherDelaysHourly - weatherDelaysHourly.shift(freq=timeDelta)
	dailySeasoned = temp.dropna(how='all', axis=0)
	dailySeasoned = dailySeasoned.interpolate()
	del timeDelta

	plt.figure()
	dailySeasoned['diff'].plot()

	# === Try to remove DAILY SEASONALITY by subtracting previous weeks's value
	timeDelta = datetime.timedelta(days=7)
	temp = weatherDelaysHourly - weatherDelaysHourly.shift(freq=timeDelta)
	weeklySeasoned = temp.dropna(how='all', axis=0)
	weeklyeasoned = dailySeasoned.interpolate()
	del timeDelta

	plt.figure()
	weeklyeasoned['diff'].plot()

	plt.show()


##################################################################################
if __name__ == "__main__":
	main()
	print('Done!')