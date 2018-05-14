import os
import pandas
import numpy
import scipy.fftpack
import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
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
	temp = delays.copy()
	temp.loc[:, 'hour_of_day'] = pandas.to_datetime(temp.time).dt.hour
	temp = temp.groupby('hour_of_day').mean() 
	fig, ax = plt.subplots(1)
	ax.plot(temp.index, temp['diff'])
	ax.set_ylabel('Average delay [s]')
	ax.set_xlabel('Time of Day [HH:MM]')

	for tick in ax.get_xticklabels():
		tick.set_rotation(90)

	plt.savefig('delay_vs_time-of-day.png')

	# === Merge with WEATHER data 
	weatherDelays = weather.merge(delays, right_on='datetime', left_index=True, how='inner')
	weatherDelays.to_csv(os.path.join(dt.data_dir(), 'weather_delays_merged.csv'))

	# ==== Remove NaN where there is no public transport data
	mask = weatherDelays.datetime > datetime.datetime(2018,2,4)
	weatherDelays = weatherDelays[mask]
	del mask
	
	cumulativeWeatherDelays = weatherDelays.groupby('datetime').mean()
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

	# === Plot data with and without seasoning treatment 
	fig, axes = plt.subplots(2, sharex=True, figsize=(15, 10))
	matplotlib.rc('xtick', labelsize=24) 
	matplotlib.rc('ytick', labelsize=24)
	axis=0
	axes[axis].plot(weeklySeasoned.index, cumulativeWeatherDelays.reindex(weeklySeasoned.index)['diff'])
	axes[axis].set_xlabel('Without de-seasoning')
	axes[axis].set_ylabel('Delay [s]')

	axis+=1
	axes[axis].plot(weeklySeasoned.index, weeklySeasoned['diff'])
	axes[axis].set_xlabel('With de-seasoning')
	axes[axis].set_ylabel('Delay [s]')

	fig.savefig('seasoned_vs_de-seasoned_delay_data.png')

	# === Plot data without de-seasoning and rainfall data
	fig, axes = plt.subplots(2, sharex=True, figsize=(15, 10))

	axis=0
	axes[axis].plot(cumulativeWeatherDelays.index, cumulativeWeatherDelays['diff'])
	axes[axis].set_xlabel('Without de-seasoning')
	axes[axis].set_ylabel('Delay [s]')

	axis+=1
	axes[axis].plot(weeklySeasoned.index, weeklySeasoned['niederschlag_mm'])
	axes[axis].set_xlabel('Rain data')
	axes[axis].set_ylabel('Rainfall [mm]')

	fig.savefig('seasoned_vs_rainfall_data.png')


	# === Plot delay-vs-weather graphs for de-seasoned data

	'''
	Description:
		Scatter plot between CUMULATIVE MM RAIN and DELAYS
	'''
	mask = cumulativeWeatherDelays.reindex(index=weeklySeasoned.index)['niederschlag_mm'] > 0 
	xData = cumulativeWeatherDelays.reindex(index=weeklySeasoned.index)['niederschlag_mm'][mask]
	yData = weeklySeasoned['diff'].loc[xData.index]
	corrMat = numpy.corrcoef(xData, yData)
	corrCoefPatch = mpatches.Patch(color='blue', label='Correlation coefficient := %.2f' %corrMat[0][1])
	plt.figure()
	plt.scatter(x=xData, y=yData, marker='x')
	plt.xlabel('Precipitation (mm)')
	plt.ylabel('De-seasoned delay (s)')
	plt.legend(handles=[corrCoefPatch])
	plt.tight_layout()
	plt.savefig('corr_rain_vs_delay_-_with_de-seasoning.png')

	del xData, yData

	mask = cumulativeWeatherDelays['niederschlag_mm'] > 0 
	xData = cumulativeWeatherDelays['niederschlag_mm'][mask]
	yData = cumulativeWeatherDelays['diff'].loc[xData.index]
	corrMat = numpy.corrcoef(xData, yData)
	corrCoefPatch = mpatches.Patch(color='blue', label='Correlation coefficient := %.2f' %corrMat[0][1])
	plt.figure()
	plt.scatter(x=xData, y=yData, marker='x')
	plt.xlabel('Precipitation [mm]')
	plt.ylabel('Delay [s]')
	plt.legend(handles=[corrCoefPatch])
	plt.tight_layout()
	plt.savefig('corr_rain_vs_delay_-_no_de-seasoning.png')

	del xData, yData

	'''
	Description:
		Time-series plot between CUMULATIVE RAIN and DE-SEASONED DELAY
	'''
	xData = cumulativeWeatherDelays['niederschlag_mm']
	print(xData)
	yData = cumulativeWeatherDelays['diff']

	fig, ax = plt.subplots(2, sharex=True, figsize=(15, 10))

	axis=0
	ax[axis].plot(yData.index, yData) 
	ax[axis].set_ylabel('Delay [s]')

	axis+=1
	ax[axis].bar(xData.index, height=xData, width=0.05, color='green')
	ax[axis].set_xlabel('YYYY-MM-DD:HH')
<<<<<<< HEAD
	ax[axis].set_ylabel('OTELFINGEN - CUM. HOURLY PRECIPITATION [mm]')
	plt.tight_layout()
=======
	ax[axis].set_ylabel('Precipitation [mm]')
>>>>>>> 75b31de49b39ce14bd7c974f3da31600ecd0ee94

	fig.savefig(os.path.join(dt.output_dir(), 'delay_vs_rainfall.png'))

	del mask, xData, yData

	'''
	Description:
		Scatter plot between AVERAGE TEMPRATURE and DELAYS
	'''
	xData = averageWeatherDelays.reindex(index=weeklySeasoned.index)['temp_degrees_c_mittel']
	yData = weeklySeasoned['diff']
	plt.figure()
	plt.scatter(x=xData, y=yData, marker='x')
	plt.xlabel('AVG TEMPERATURE [C]')
	plt.ylabel('DE-SEASONED DELAY [s]')
	plt.tight_layout()

	'''
	Description:
		Time-series plot between AVERAGE TEMPERATURE and DE-SEASONED DELAY
	'''
	fig, ax = plt.subplots(2, sharex=True)

	axis=0
	ax[axis].plot(yData.index, yData) 
	ax[axis].set_ylabel('DE-SEASONED DELAY [s]')

	axis+=1
	ax[axis].bar(xData.index, height=xData, width=0.14, color='red')
	ax[axis].set_xlabel('YYYY-MM-DD:HH')
	ax[axis].set_ylabel('AVG. HOURLY TEMPERATURE [C]')
	plt.tight_layout()


	del xData, yData
	

	plt.show()


##################################################################################
def analyze_weather_delays():

	logger = logging.getLogger(__name__)

	# === Read DELAYS and WEATHER data
	logger.info('<--Fetching data-->')
	delays = dt.get_lineie_69_data()
	flat = dt.get_linie_94_data()
	weather = dt.get_iac_weather_data()
	oldWeather = dt.get_weather_data()

	# === Check for outliers/errors in weather data 
	q = 3 #weather.rain.quantile(0.99975)
	mask = weather.rain < q
	weather = weather[mask]
	del mask, q

	# === Focus on BUS 69
	logger.info('<--Prepare bus 69 data-->')
	mask = delays.linie == 69
	delays = delays[mask]
	delays.reset_index(drop=True, inplace=True)
	del mask

	# ==== Remove NaN where there is no public transport data
	mask = delays.betriebsdatum > datetime.datetime(2018, 2, 4)
	delays = delays[mask]
	del mask

	# ==== Flat line - bus 94
	logger.info('<--Prepare bus 94 delay data-->')
	flat.loc[:, 'diff'] = flat.ist_an_von - flat.soll_an_von
	flat.loc[:, 'time'] = pandas.to_datetime(flat.soll_an_von.copy().astype(float), errors='coerce', unit='s')
	flat.time = flat.time.dt.strftime('%H:%M')
	flat.loc[:, 'datetime'] = pandas.to_datetime(flat.datum_von.astype(str) + ' ' + flat.time)
	flat.datetime = flat.datetime.dt.round('60min')

	# === Extract exact time delays
	logger.info('<--Prepare bus 69 delay data-->')
	delays.loc[:, 'diff'] = delays.ist_an_von - delays.soll_an_von
	delays.loc[:, 'time'] = pandas.to_datetime(delays.soll_an_von.copy().astype(float), errors='coerce', unit='s')
	delays.time = delays.time.dt.strftime('%H:%M')
	delays.loc[:, 'datetime'] = pandas.to_datetime(delays.datum_von.astype(str) + ' ' + delays.time)
	delays.datetime = delays.datetime.dt.round('60min')

	# === Try to remove DAILY SEASONALITY by subtracting previous weeks's value
	logger.info('<--Compute de-seasoning for bus lines-->')
	_delays = delays.set_index('datetime', drop=True)
	_delays.index = pandas.to_datetime(_delays.index)
	__delays = _delays.groupby(_delays.index).sum()

	_flat = flat.set_index('datetime', drop=True)
	_flat.index = pandas.to_datetime(_flat.index)
	__flat = _flat.groupby(_flat.index).sum()

	timeDelta = datetime.timedelta(days=7)
	temp = __delays['diff'].copy() - __delays['diff'].shift(freq=timeDelta)
	weeklyDetrendedBus69 = temp.dropna(how='all', axis=0)
	weeklyDetrendedBus69 = weeklyDetrendedBus69.interpolate()

	temp = __flat['diff'].copy() - __flat['diff'].shift(freq=timeDelta)
	weeklyDetrendedBus94 = temp.dropna(how='all', axis=0)
	weeklyDetrendedBus94 = weeklyDetrendedBus94.interpolate()

	del timeDelta, temp

	plt.figure()
	weeklyDetrendedBus69.plot(title='de-seasoned delay (bus 69) data (diff-of-diff)')

	plt.figure()
	weeklyDetrendedBus94.plot(title='de-seasoned delay (bus 94) data (diff-of-diff)')

	plt.show()

	# === Extract weather measures
	weather.loc[:, 'datetime'] = weather.index.round('60min')

	# === GROUPBY and RESAMPLE 
	groupSumDelaysByHour = delays.groupby('datetime').sum()
	groupMeanDelaysByHour = delays.groupby('datetime').mean()
	groupSumFlatByHour = flat.groupby('datetime').sum()
	groupMeanFlatByHour = flat.groupby('datetime').mean()
	resampleSumWeatherByHour = weather.resample('H').sum()
	resampleMeanWeatherByHour = weather.resample('H').mean()

<<<<<<< HEAD
	# === Plot precipitation by hour
	fig, ax = plt.subplots(1, figsize=(15, 10))
	ax.bar(resampleSumWeatherByHour.index, height=resampleSumWeatherByHour, width=0.05, color='green')
	ax.set_xlabel('YYYY-MM-DD:HH')
	ax.set_ylabel('HOENGGERBERG - CUM. HOURLY PRECIPITATION [mm]')
	plt.tight_layout()

	fig.savefig('precipitation_hoengg.png')
    
    # === Plot delays vs. weather data

	print(xData)
	xData = resampleSumWeatherByHour.rain
	yData = cumulativeWeatherDelays['diff']

	fig, ax = plt.subplots(2, sharex=True, figsize=(15, 10))

	axis=0
	ax[axis].plot(yData.index, yData) 
	ax[axis].set_ylabel('Delay [s]')

	axis+=1
	ax[axis].bar(xData.index, height=xData, width=0.05, color='green')
	ax[axis].set_xlabel('YYYY-MM-DD:HH')
	ax[axis].set_ylabel('HOENGGERBERG - CUM. HOURLY PRECIPITATION [mm]')
	plt.tight_layout()

	fig.savefig('delay_vs_rainfall.png')

	del mask, xData, yData
    
=======
	resampleMeanWeatherByHour.to_csv(os.path.join(dt.output_dir(), 'newWeatherData.csv'))

>>>>>>> 75b31de49b39ce14bd7c974f3da31600ecd0ee94
	# === Feature transformations
	maskSnow = resampleMeanWeatherByHour.T_air < 0
	feature = resampleMeanWeatherByHour.rain * maskSnow.astype(int) 

	fig, ax = plt.subplots(6, sharex=True)

	window = 24

	axis = 0
	resampleSumWeatherByHour.rain.plot(ax=ax[axis])
	ax[axis].set_ylabel('[mm]')

	axis += 1
	pandas.rolling_mean(resampleSumWeatherByHour.rain, window).plot(ax=ax[axis])
	ax[axis].set_ylabel('[mm]' )

	axis += 1
	weeklyDetrendedBus69.plot(ax=ax[axis], color='orange')
	ax[axis].set_ylabel('[s]')
	
	axis += 1
	weeklyDetrendedBus69.rolling(window=6).mean().plot(ax=ax[axis], color='orange')
	ax[axis].set_ylabel('[s]')

	axis += 1
	feature.plot(ax=ax[axis], color='green')
	ax[axis].set_ylabel('[mm]')
	
	axis+=1
	feature.rolling(window=6).mean().plot(ax=ax[axis], color='green')
	ax[axis].set_ylabel('[mm]')

	plt.show()

	fig, ax = plt.subplots(1)
	resampleSumWeatherByHour.rain.plot(ax=ax, color='green', title='Weather Data - Hoenggerberg Station')
	ax.set_ylabel('Precipitation [mm]')

	plt.show()

	# === Combine features into new data-frame
	logger.info('<--Construct new features - bus 69-->')
	combine = [
		weeklyDetrendedBus69.rolling(window=6).mean(),
		resampleSumWeatherByHour.rain.rolling(window=window).mean(),
		pandas.Series(feature.rolling(window=6).mean(), name='snow'),
	]

	df = pandas.concat(combine, axis=1).dropna(how='any')
	print(df.corr())

	# === Scatter plots for rain and snow
	mask = df['diff'] > 0 
	_df = df.copy()[mask]
	corr = _df.corr()
	fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
	axis=0
	ax[axis].scatter(y=_df['diff'], x=_df['rain'], marker='x', color='blue')
	ax[axis].set_xlabel('precipitation [mm]')
	axis+=1
	ax[axis].scatter(y=_df['diff'], x=_df['snow'], marker='x', color='green')
	ax[axis].set_xlabel('snow [mm]')

	corrDelayRain = mpatches.Patch(color='blue', label='Cum. delay [s] vs. precipitation [mm] - correlation %.4f' %(corr.loc['diff','rain']))
	corrDelaySnow = mpatches.Patch(color='green', label='Cum. delay [s] vs. snow [mm] - correlation %.4f' %(corr.loc['diff','snow']))
	plt.legend(handles=[corrDelayRain, corrDelaySnow])

	plt.savefig(os.path.join(dt.output_dir(), 'bus_69_correlation_precipitation_and_snow.png')) 
	print('Correlation between delay, rain, snow. Delay>0')
	print(corr)

	# === Combine features into new data-frame
	logger.info('<--Construct new features - bus 94-->')
	combine = [
		weeklyDetrendedBus94.rolling(window=6).mean(),
		resampleSumWeatherByHour.rain.rolling(window=window).mean(),
		pandas.Series(feature.rolling(window=6).mean(), name='snow'),
	]

	df = pandas.concat(combine, axis=1).dropna(how='any')
	print(df.corr())

	# === Scatter plots for rain and snow
	mask = df['diff'] > 0 
	_df = df.copy()[mask]
	corr = _df.corr()
	fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
	axis=0
	ax[axis].scatter(y=_df['diff'], x=_df['rain'], marker='x', color='blue')
	ax[axis].set_xlabel('precipitation [mm]')
	axis+=1
	ax[axis].scatter(y=_df['diff'], x=_df['snow'], marker='x', color='green')
	ax[axis].set_xlabel('snow [mm]')

	corrDelayRain = mpatches.Patch(color='blue', label='Cum. delay [s] vs. precipitation [mm] - correlation %.4f' %(corr.loc['diff','rain']))
	corrDelaySnow = mpatches.Patch(color='green', label='Cum. delay [s] vs. snow [mm] - correlation %.4f' %(corr.loc['diff','snow']))
	plt.legend(handles=[corrDelayRain, corrDelaySnow])

	plt.savefig(os.path.join(dt.output_dir(), 'bus_94_correlation_precipitation_and_snow.png'))	
	print('Correlation between delay, rain, snow. Delay>0')
	print(corr)

	# === Corr between std. diff and rain 
	df = resampleSumWeatherByHour.join(weeklyDetrendedBus69, how='left')
	print(df.corr().loc['diff', 'rain'])


##################################################################################
def analyze_all_bus_lines():

	import math

	from sklearn.preprocessing import OneHotEncoder
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score
	from sklearn.metrics import mean_squared_error

	logger = logging.getLogger(__name__)

	## 
	logger.info('<--Read bus delay data-->')
	zvv = pandas.read_hdf(os.path.join('data', 'zvv_all_bus_lines.h5'))

	##
	logger.info('<--Read weather data & adjust for outliers-->')
	weather = dt.get_iac_weather_data()

	q = 3 #weather.rain.quantile(0.99975)
	mask = weather.rain < q
	weather = weather[mask]
	del mask, q

	##
	logger.info('<--Pre-process bus delay data-->')
	zvv.loc[:, 'diff'] = zvv.ist_an_von - zvv.soll_an_von
	zvv.loc[:, 'time'] = pandas.to_datetime(zvv.soll_an_von.astype(float), errors='coerce', unit='s')
	zvv.time = zvv.time.dt.strftime('%H:%M')
	zvv.loc[:, 'datetime'] = pandas.to_datetime(zvv.datum_von.astype(str) + ' ' + zvv.time)
	zvv.datetime = zvv.datetime.dt.round('60min')

	##
	logger.info('<--Extract weather measures-->')
	weather.loc[:, 'datetime'] = weather.index.round('60min')

	resampleSumWeatherByHour = weather.resample('H').sum()
	resampleMeanWeatherByHour = weather.resample('H').mean()

	maskSnow = resampleMeanWeatherByHour.T_air < 0
	feature = resampleMeanWeatherByHour.rain * maskSnow.astype(int) 

	##
	logger.info('<--Compute de-seasoning for all bus lines-->')
	container = []
	for line in numpy.sort(numpy.setdiff1d(zvv.linie.unique(), [753, 29])):

		##
		logger.info('<--Compute groupby sum on datetime for all bus line %i -->' %line)
		transport = zvv[zvv.linie == line]
		transport.set_index('datetime', drop=True, inplace=True)
		transport.index = pandas.to_datetime(transport.index)
		transport = transport.groupby(transport.index).sum()

		timeDelta = datetime.timedelta(days=7)
		temp = transport['diff'].copy() - transport['diff'].shift(freq=timeDelta)
		weeklyDetrendedBus = temp.dropna(how='all', axis=0)
		weeklyDetrendedBus = weeklyDetrendedBus.interpolate()
		del timeDelta, temp

		##
		logger.info('<--Combine line %i features into new data-frame-->' %line)
		window = 6
		combine = [
			weeklyDetrendedBus.rolling(window=window).mean(),
			resampleSumWeatherByHour.rain.rolling(window=window).mean(),
			pandas.Series(feature.rolling(window=window).mean(), name='snow'),
			pandas.Series(resampleMeanWeatherByHour.T_air.rolling(window=window).mean(), name='temp')
		]

		df = pandas.concat(combine, axis=1).dropna(how='any')
		df.loc[:, 'weekday'] = df.index.dayofweek
		df.loc[:, 'hour'] = df.index.hour

		mask = (df['diff'] > 0)
		df = df[mask]
		corr = df.corr()

		logger.info('<--1. Categorical features -> one-hot encoder-->')
		data = df.sort_values(['weekday', 'hour'])
		encoder = OneHotEncoder()
		
		categoricalFeatures = [
			'weekday', 
			'hour'
		]

		encoderFeatureOrder = [
			*data.weekday.unique(),
			*data.hour.unique(),
		]

		enc = encoder.fit(data.loc[:, categoricalFeatures])
		categoricalData = enc.transform(data.loc[:, categoricalFeatures])

		logger.info('<--2. Ordinal features -> no transform -->')
		target = ['diff']
		data.drop(columns=categoricalFeatures)
		ordinalFeatures = data.columns.difference(target)


		logger.info('<--2. Regression-->')
		trainX = numpy.hstack([data.loc[:, ordinalFeatures].values, categoricalData.todense()])
		trainY = data.loc[:, target].values.flatten()
		reg = LinearRegression(fit_intercept=True)
		reg.fit(X=trainX, y=trainY)
		predict = reg.predict(X=trainX)

		logger.info('<--3. Results & Plot-->')
		a, b = numpy.polyfit(trainY, predict, deg=1)
		f = lambda x: a*x + b

		fig, ax = plt.subplots(1)
		ax.scatter(y=trainY, x=predict, color='red', marker='x')
		ax.plot(predict, f(predict))
		ax.set_aspect('equal')
		ax.grid(True)
		ax.set_ylabel('Actual - delay')
		ax.set_xlabel('Predicted - delay')
		ax.set_title('Linear Regression Model - Line %i' %line)

		r2 = r2_score(trainY, predict)
		mse = mean_squared_error(trainY, predict)

		corrDelayRain = mpatches.Patch(color='blue', label='R^2 %.4f' %r2)
		corrDelaySnow = mpatches.Patch(color='blue', label='RMSE %.4f' %math.sqrt(mse))
		plt.legend(handles=[corrDelayRain, corrDelaySnow])

		plt.savefig(os.path.join(dt.output_dir(), 'line_%i_prediction.png' %line))

		logger.info('<--4. Correlation structure-->')
		print(df.corr())

		logger.info('<--5. Save summary statistics to file-->')
		stats = pandas.Series(data=corr.loc['diff',:], name=line)
		stats['r2'] = r2
		stats['mse'] = mse
		container.append(stats)


	pandas.concat(container, axis=1).to_csv(os.path.join(dt.output_dir(), 'correlation_all_bus_lines.csv'))



##################################################################################
def analyze_all_tram_lines():

	import math

	from sklearn.preprocessing import OneHotEncoder
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import r2_score
	from sklearn.metrics import mean_squared_error

	logger = logging.getLogger(__name__)

	## 
	logger.info('<--Read bus delay data-->')
	zvv = pandas.read_hdf(os.path.join('data', 'zvv_all_tram_lines.h5'))

	##
	logger.info('<--Read weather data & adjust for outliers-->')
	weather = dt.get_iac_weather_data()

	q = 3 #weather.rain.quantile(0.99975)
	mask = weather.rain < q
	weather = weather[mask]
	del mask, q

	##
	logger.info('<--Pre-process tram delay data-->')
	zvv.loc[:, 'diff'] = zvv.ist_an_von - zvv.soll_an_von
	zvv.loc[:, 'time'] = pandas.to_datetime(zvv.soll_an_von.astype(float), errors='coerce', unit='s')
	zvv.time = zvv.time.dt.strftime('%H:%M')
	zvv.loc[:, 'datetime'] = pandas.to_datetime(zvv.datum_von.astype(str) + ' ' + zvv.time)
	zvv.datetime = zvv.datetime.dt.round('60min')

	##
	logger.info('<--Extract weather measures-->')
	weather.loc[:, 'datetime'] = weather.index.round('60min')

	resampleSumWeatherByHour = weather.resample('H').sum()
	resampleMeanWeatherByHour = weather.resample('H').mean()

	maskSnow = resampleMeanWeatherByHour.T_air < 0
	feature = resampleMeanWeatherByHour.rain * maskSnow.astype(int) 

	##
	logger.info('<--Compute de-seasoning for all tram lines-->')
	container = []
	for line in numpy.sort(numpy.setdiff1d(zvv.linie.unique(), [753, 29])):

		##
		logger.info('<--Compute groupby sum on datetime for all tram line %i -->' %line)
		transport = zvv[zvv.linie == line]
		transport.set_index('datetime', drop=True, inplace=True)
		transport.index = pandas.to_datetime(transport.index)
		transport = transport.groupby(transport.index).sum()

		timeDelta = datetime.timedelta(days=7)
		temp = transport['diff'].copy() - transport['diff'].shift(freq=timeDelta)
		weeklyDetrendedtram = temp.dropna(how='all', axis=0)
		weeklyDetrendedtram = weeklyDetrendedtram.interpolate()
		del timeDelta, temp

		##
		logger.info('<--Combine line %i features into new data-frame-->' %line)
		window = 6
		combine = [
			weeklyDetrendedtram.rolling(window=window).mean(),
			resampleSumWeatherByHour.rain.rolling(window=window).mean(),
			pandas.Series(feature.rolling(window=window).mean(), name='snow'),
			pandas.Series(resampleMeanWeatherByHour.T_air.rolling(window=window).mean(), name='temp')
		]

		df = pandas.concat(combine, axis=1).dropna(how='any')
		df.loc[:, 'weekday'] = df.index.dayofweek
		df.loc[:, 'hour'] = df.index.hour

		mask = (df['diff'] > 0)
		df = df[mask]
		corr = df.corr()

		logger.info('<--1. Categorical features -> one-hot encoder-->')
		data = df.sort_values(['weekday', 'hour'])
		encoder = OneHotEncoder()
		
		categoricalFeatures = [
			'weekday', 
			'hour'
		]

		encoderFeatureOrder = [
			*data.weekday.unique(),
			*data.hour.unique(),
		]

		enc = encoder.fit(data.loc[:, categoricalFeatures])
		categoricalData = enc.transform(data.loc[:, categoricalFeatures])

		logger.info('<--2. Ordinal features -> no transform -->')
		target = ['diff']
		data.drop(columns=categoricalFeatures)
		ordinalFeatures = data.columns.difference(target)


		logger.info('<--2. Regression-->')
		trainX = numpy.hstack([data.loc[:, ordinalFeatures].values, categoricalData.todense()])
		trainY = data.loc[:, target].values.flatten()
		reg = LinearRegression(fit_intercept=True)
		reg.fit(X=trainX, y=trainY)
		predict = reg.predict(X=trainX)

		logger.info('<--3. Results & Plot-->')
		a, b = numpy.polyfit(trainY, predict, deg=1)
		f = lambda x: a*x + b

		fig, ax = plt.subplots(1)
		ax.scatter(y=trainY, x=predict, color='red', marker='x')
		ax.plot(predict, f(predict))
		ax.set_aspect('equal')
		ax.grid(True)
		ax.set_ylabel('Actual - delay')
		ax.set_xlabel('Predicted - delay')
		ax.set_title('Linear Regression Model - Line %i' %line)

		r2 = r2_score(trainY, predict)
		mse = mean_squared_error(trainY, predict)

		corrDelayRain = mpatches.Patch(color='blue', label='R^2 %.4f' %r2)
		corrDelaySnow = mpatches.Patch(color='blue', label='RMSE %.4f' %math.sqrt(mse))
		plt.legend(handles=[corrDelayRain, corrDelaySnow])

		plt.savefig(os.path.join(dt.output_dir(), 'line_%i_prediction.png' %line))

		logger.info('<--4. Correlation structure-->')
		print(df.corr())

		logger.info('<--5. Save summary statistics to file-->')
		stats = pandas.Series(data=corr.loc['diff',:], name=line)
		stats['r2'] = r2
		stats['mse'] = mse
		container.append(stats)


	pandas.concat(container, axis=1).to_csv(os.path.join(dt.output_dir(), 'correlation_all_tram_lines.csv'))


##################################################################################
if __name__ == "__main__":
<<<<<<< HEAD
	main()
	#analyze_weather_delays()
=======

	import logging
	import sys

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)	

	font = {'family' : 'normal',
	        'weight' : 'bold',
	        'size'   : 14}

	matplotlib.rc('font', **font)	

	# main()
	# analyze_weather_delays()
	analyze_all_bus_lines()
	analyze_all_tram_lines()
>>>>>>> 75b31de49b39ce14bd7c974f3da31600ecd0ee94
	print('Done!')