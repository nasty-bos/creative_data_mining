import pandas
import numpy

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

	# === Eliminate DAILY SEASONALITY by subtracting previous day's value

	plt.show()


##################################################################################
if __name__ == "__main__":
	main()
	print('Done!')