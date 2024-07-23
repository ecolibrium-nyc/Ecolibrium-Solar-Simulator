import pandas as pd
import math
from nyisotoolkit import NYISOData, NYISOStat, NYISOVis
from classes import *
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

#Weather Data
originalweatherdf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\10002 2023-07-18 to 2024-07-16.csv') #read weather csv file
originalweatherdf['sunrise'] = pd.to_datetime(originalweatherdf['sunrise']) #cnvert to datetime format
originalweatherdf['sunset'] = pd.to_datetime(originalweatherdf['sunset'])
originalweatherdf['day_length'] = originalweatherdf['sunset'] - originalweatherdf['sunrise'] #calc day length
originalweatherdf = originalweatherdf.drop(columns=['name', 'description', 'stations']) #remove columns

originalweatherdf['solarenergy'] = originalweatherdf['solarenergy'] * 227.78 #convert MJ/m^2 to wh/m^2
originalweatherdf = originalweatherdf[originalweatherdf['solarenergy'] != 0] #remove all days w/ unmeasured solar energies(21)
originalweatherdf.reset_index(drop=True, inplace=True)

AVGLD = originalweatherdf['day_length'].mean()
AVGSE = originalweatherdf['solarenergy'].mean()

print('AVG Solar Energy (wh/m^2) for the year ' + str(AVGSE))
print('AVG Day Length for the year ' + str(AVGLD))

#Building Data

buildingdf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Building Data.csv') #read Building csv file
Buildings = {}
for index, row in buildingdf.iterrows():
    address = row['Address']
    units = row['Number of Units']
    TheoSpace = row['Theoretical Space for Panels (m^2)']
    UsagePproperty = row['Average usage per unit per day (w)']
    numberofbatteries = row['Number of Batteries in Building']
    storagePbattery = row['Storage Size Per Battery (Watts)']
    building = Building(units, TheoSpace, UsagePproperty, numberofbatteries, storagePbattery)
    Buildings[address] = building


#NYISO DATA
loaddf = NYISOData(dataset='load_h', year='2023').df # year argument in local time, but returns dataset in UTC
otherloaddf =  NYISOData(dataset='load_h', year='2024').df # year argument in local time, but returns dataset in UTC
loaddf = pd.concat([loaddf, otherloaddf], ignore_index=False)
loaddf = loaddf.loc[:, ['N.Y.C.']]
loaddf = loaddf.reset_index()
loaddf.columns = ['Time', 'N.Y.C.']

#Match Weather Data to NYISO Data
originalweatherdf['datetime'] = pd.to_datetime(originalweatherdf['datetime']).dt.date
loaddf['Time'] = pd.to_datetime(loaddf['Time'])
loaddf['Hour'] = loaddf['Time'].dt.hour
loaddf['saved_hour'] = loaddf['Hour']
loaddf = loaddf[loaddf['Time'].dt.date.isin(originalweatherdf['datetime'])]
loaddf['Time'] = loaddf.apply(
    lambda row: pd.to_datetime(row['Time'].date()) + pd.DateOffset(hours=row['saved_hour']), axis=1
)
loaddf = loaddf.drop(columns=['Hour', 'saved_hour'])
loaddf['N.Y.C.'] = loaddf['N.Y.C.'] * 1000000 #convert MW to W

#set hourly and daily load df
loadhourly = loaddf.copy()
loaddaily = loaddf.copy()

#set date as the index
loaddf.set_index('Time', inplace=True)

#Convert to daily structure
loaddaily['Time'] = pd.to_datetime(loaddaily['Time'])
loaddaily['Time'] = loaddaily['Time'].dt.date
loaddaily = loaddaily.groupby('Time')['N.Y.C.'].sum().reset_index()
TloadBefore = loaddaily['N.Y.C.'].sum()


#Weather DF
originalweatherdf['WeatherClass'] = Weather(originalweatherdf['day_length'], originalweatherdf['preciptype'], originalweatherdf['temp'])
Weatherdf = originalweatherdf['WeatherClass']
originalweatherdf = originalweatherdf.drop(columns=['WeatherClass'])
distributionweatherdf = originalweatherdf.copy()

#Hourly Data Distribution

#import everything
Hspringdf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Spring Hourly 0407-0411.csv')
Hsummerdf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Summer Hourly 0701-0716.csv')
Hwinterdf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Winter Hourly 0111-0126.csv')
Hfalldf = pd.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Fall 1026-1029.csv')
# Concatenate hourly dataframes
Hourlydf = pd.concat([Hspringdf, Hsummerdf, Hwinterdf, Hfalldf], ignore_index=True)
# Extract sunrise, sunset, and daily solar energy for the year
distributionweatherdf['datetime'] = pd.to_datetime(distributionweatherdf['datetime'])
distributionweatherdf.set_index('datetime', inplace=True)
sunrise_sunset_df = distributionweatherdf[['sunrise', 'sunset', 'solarenergy']].resample('D').first()
# Convert 'sunrise' and 'sunset' to hours
sunrise_sunset_df['sunrise_hour'] = pd.to_datetime(sunrise_sunset_df['sunrise'], format='%H:%M').dt.hour + pd.to_datetime(sunrise_sunset_df['sunrise'], format='%H:%M').dt.minute / 60
sunrise_sunset_df['sunset_hour'] = pd.to_datetime(sunrise_sunset_df['sunset'], format='%H:%M').dt.hour + pd.to_datetime(sunrise_sunset_df['sunset'], format='%H:%M').dt.minute / 60
sunrise_sunset_df.dropna(subset=['sunrise_hour', 'sunset_hour'], inplace=True)
# Generate hourly statistics from the concatenated data
hourly_stats = Hourlydf.groupby('datetime')['solarenergy'].agg(['mean', 'std']).reset_index()
# Create hourly distributions
hourly_distributions = {
    row['datetime']: norm(loc=row['mean'], scale=row['std']) for _, row in hourly_stats.iterrows()
}
# Function to smooth data
def smooth_data(data, sigma=1):
    return gaussian_filter1d(data, sigma=sigma)
# Function to generate hourly data
def generate_hourly_data(date, total_energy, sunrise_hour, sunset_hour, hourly_distributions):
    hourly_energy = []
    for hour in range(24):
        dist = hourly_distributions.get(hour, norm(loc=0, scale=1))
        energy = dist.rvs()
        energy = max(energy, 0)  # Ensure no negative values
        hourly_energy.append(energy)
    # Apply smoothing
    hourly_energy = smooth_data(np.array(hourly_energy))
    # Adjust based on daylight hours and realistic solar pattern
    daylight_mask = (np.arange(24) >= sunrise_hour) & (np.arange(24) <= sunset_hour)
    hours = np.arange(24)
    peak_hour = (sunrise_hour + sunset_hour) / 2
    solar_pattern = np.clip(np.cos((hours - peak_hour) * np.pi / (sunset_hour - sunrise_hour)), 0, None)
    hourly_energy = hourly_energy * solar_pattern
    # Scale the hourly data to match the total energy for the day
    hourly_energy = hourly_energy / hourly_energy.sum() * total_energy
    return pd.DataFrame({'date': date, 'hour': range(24), 'solarenergy': hourly_energy})
# Generate hourly data for the entire year
hourly_data_list = []
for idx, row in sunrise_sunset_df.iterrows():
    daily_hourly_data = generate_hourly_data(row.name, row['solarenergy'], row['sunrise_hour'], row['sunset_hour'], hourly_distributions)
    hourly_data_list.append(daily_hourly_data)
# Concatenate all the generated hourly data
hourly_data_df = pd.concat(hourly_data_list).reset_index(drop=True)
# Print the resulting DataFrame
#print(hourly_data_df)

#Main Project
Blevel = 0 #represented as a percentage(.1 = 10%)
SUsed = 0
GUsed = 0
iterations=0
"""Buildings = {'Address1': Building(100, 2500, 10000, 200, 5000),
             'Address2': Building(50, 2500, 10000, 100, 5000),
             'Address3': Building(200, 2500, 10000, 400, 5000),
             'Address4': Building(1000, 2500, 10000, 2000, 5000)}"""
BasePanel = SolarPanel(500, .21, .05, .003)
hourly_data_df['date'] = pd.to_datetime(hourly_data_df['date'])
originalweatherdf['datetime'] = pd.to_datetime(originalweatherdf['datetime'])
originalweatherdf.set_index('datetime', inplace=True)
#fix hourly_data_df
hourly_data_df['date'] = hourly_data_df.apply(lambda row: row['date'].replace(hour=row['hour']), axis=1)

for index, row in hourly_data_df.iterrows():
  date = row['date']
  hourly_solar = row['solarenergy']
  energyproduced = calcsolarpperiod(hourly_solar, BasePanel, originalweatherdf, date) #take into account solar panel efficiencies
  for address in Buildings:
      energy_stored = 0
      energy_needed = (get_hourly_usage(date.hour,Buildings[address].units * Buildings[address].UsagePproperty))
      energy_obtained = Buildings[address].TheoSpace * energyproduced

      # Prioritize battery storage
      if Buildings[address].BatteryLevel < 1:
          remainingstorage = (1 - Buildings[address].BatteryLevel) * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
          if energy_obtained >= remainingstorage:
              energy_stored = remainingstorage
              Buildings[address].BatteryLevel = 1
          else:
              energy_stored = energy_obtained
              Buildings[address].BatteryLevel += energy_stored / (Buildings[address].storagePbattery * Buildings[address].numberofbatteries)
          energy_obtained -= energy_stored

      # Use stored energy if solar energy is insufficient
      if energy_obtained < energy_needed:
          energy_needed -= energy_obtained
          if  Buildings[address].BatteryLevel > 0:
              if Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries >= energy_needed:
                  Buildings[address].BatteryLevel -= energy_needed / (Buildings[address].storagePbattery * Buildings[address].numberofbatteries)
                  SUsed += energy_needed
                  loaddf.at[date, 'N.Y.C.'] -= energy_needed
                  energy_needed = 0
              else:
                  SUsed += Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  loaddf.at[date, 'N.Y.C.'] -= Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  energy_needed -= Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  Buildings[address].BatteryLevel = 0
          GUsed += energy_needed
      else:
          SUsed += energy_needed
          loaddf.at[date, 'N.Y.C.'] -= energy_needed
  iterations +=1

print(iterations)
print(f'Total Solar Energy Used:       {SUsed} Wh')
print(f'Total Grid Energy Used:        {GUsed} Wh')
print(f'Total Energy Used:             {SUsed+GUsed} Wh')
usageongrid = 0
for x in Buildings:
  usageongrid +=(Buildings[x].units * Buildings[x].UsagePproperty)/24 # usagephour
expected_total_energy_used = iterations * usageongrid  # number of hours iterated * usage on whole grid per hour
print(f'Expected Total Energy Used:    {expected_total_energy_used} Wh')
TLoad = loaddf['N.Y.C.'].sum()
print(f'Total load in NYC befor solar: {TloadBefore} wh')
print(f'Total load in NYC after Solar: {TLoad} wh')