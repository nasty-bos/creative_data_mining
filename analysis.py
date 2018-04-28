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
	
	cumulativeWeatherDelays = weatherDelays.groupby('datetime').sum()
	averageWeatherDelays = weatherDelays.groupby('datetime').mean()

	# === Estimate DAILY SEASONALITY using Fourier transform
	'''
	Description:
		Fourier transform of time-series data in time domain (yt, xt) to frequency domain (yf, xf):

	Arguments:
		:param n: (float) number of data points / observations  
		:param T: (float) maximum frequency of data i.e. 1H, 1m, 1s 	
	'''
	n, m = cumulativeWeatherDelays.shape
	T = 1/n
	yf = scipy.fftpack.fft(weatherDelays['diff'].values)
	xf = numpy.linspace(0, 1/2.0 * T, n/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/n * numpy.abs(yf[:n//2]))

	# TODO Buiild function that 1. Picks data window, 2. FFT, 3. Removes frequency in xf domain, 4. IFFT, 5. Corr

	# === Try to remove DAILY SEASONALITY by subtracting previous day's value
	timeDelta = datetime.timedelta(days=1)
	temp = cumulativeWeatherDelays.copy() - cumulativeWeatherDelays.shift(freq=timeDelta)
	dailySeasoned = temp.dropna(how='all', axis=0)
	dailySeasoned = dailySeasoned.interpolate()
	del timeDelta, temp

	plt.figure()
	dailySeasoned['diff'].plot()

	# === Try to remove DAILY SEASONALITY by subtracting previous weeks's value
	timeDelta = datetime.timedelta(days=7)
	temp = cumulativeWeatherDelays.copy() - cumulativeWeatherDelays.shift(freq=timeDelta)
	weeklySeasoned = temp.dropna(how='all', axis=0)
	weeklySeasoned = weeklySeasoned.interpolate()
	del timeDelta, temp

	plt.figure()
	weeklySeasoned['diff'].plot()

	# === Plot delay-vs-weather graphs for de-seasoned data

	'''
	Description:
		Scatter plot between CUMULATIVE MM RAIN and DELAYS
	'''
	mask = cumulativeWeatherDelays.reindex(index=weeklySeasoned.index)['niederschlag_mm'] > 0 
	xData = cumulativeWeatherDelays.reindex(index=weeklySeasoned.index)['niederschlag_mm'][mask]
	yData = weeklySeasoned['diff'].loc[xData.index]
	plt.figure()
	plt.scatter(x=xData, y=yData)
	plt.xlabel('CUMULATIVE RAINFALL in MM')
	plt.ylabel('DE-SEASONED DELAY')
	plt.tight_layout()

	del xData, yData

	'''
	Description:
		Time-series plot between CUMULATIVE RAIN and DE-SEASONED DELAY
	'''
	xData = cumulativeWeatherDelays.reindex(index=weeklySeasoned.index)['niederschlag_mm']
	yData = weeklySeasoned['diff']

	fig, ax = plt.subplots(2, sharex=True)
	axis=0
	ax[axis].plot(yData.index, yData) 
	axis+=1
	ax[axis].bar(xData.index, height=xData, width=0.05)

	del mask, xData, yData

	'''
	Description:
		Scatter plot between AVERAGE TEMPRATURE and DELAYS
	'''
	xData = averageWeatherDelays.reindex(index=weeklySeasoned.index)['temp_degrees_c_mittel']
	yData = weeklySeasoned['diff']
	plt.figure()
	plt.scatter(x=xData, y=yData, marker='x')
	plt.xlabel('AVERAGE TEMPERATURE in DEGREES CELCIUS')
	plt.ylabel('DE-SEASONED DELAY')
	plt.tight_layout()

	'''
	Description:
		Time-series plot between AVERAGE TEMPERATURE and DE-SEASONED DELAY
	'''
	fig, ax = plt.subplots(2, sharex=True)
	axis=0
	ax[axis].plot(yData.index, yData) 
	axis+=1
	ax[axis].bar(xData.index, height=xData, width=0.14)

	del xData, yData
	

	plt.show()


##################################################################################
if __name__ == "__main__":
	main()
	print('Done!')