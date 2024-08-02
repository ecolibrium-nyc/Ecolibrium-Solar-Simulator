import pandas as pd
import math
from nyisotoolkit import NYISOData, NYISOStat, NYISOVis
from classes import *
import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import time
import tkinter as tk
from packageinistaller import *

#option to install packages or run file seperately
#installpackages()

start_time = time.time()
print(f'start time: {start_time}')

weatherpath = r'10002 2023-07-18 to 2024-07-16.csv'
buildingpath = r'Building Data.csv'
FallPath = r'Fall 1026-1029.csv'
Springpath = r'Spring Hourly 0407-0411.csv'
Summerpath = r'Summer Hourly 0701-0716.csv'
Winterpath = r'Winter Hourly 0111-0126.csv'
loadsavepath = r'NewLoadpostsolar.csv'
#Weather Data
originalweatherdf = pd.read_csv(weatherpath) #read weather csv file
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

buildingdf = pd.read_csv(buildingpath) #read Building csv file
Buildings = {}
for index, row in buildingdf.iterrows():
    address = row['Address']
    units = row['Number of Units']
    TheoSpace = row['Theoretical Space for Panels (m^2)']
    TotalSquareFeet = row['Total Square Feet']
    UsagePproperty = row['Average usage per unit per day (w)']
    numberofbatteries = row['Number of Batteries in Building']
    storagePbattery = row['Storage Size Per Battery (Watts)']
    MaxCharge = row['Base Max Charge Rate (wh)']
    MaxDischarge = row['Base Max Discharge Rate (wh)']
    BatteryRoomTemp = row['Average Battery Room Temperature (F)']
    BatteryRoomHumid = row['Average Battery Room Humidity (percentage)']
    BatteryAge = row['Battery Age (y)']
    InvertEfficiency = row['Inverter Efficiency (percentage)']
    building = Building(units, TheoSpace, TotalSquareFeet, UsagePproperty, numberofbatteries, 
    storagePbattery, MaxCharge , MaxDischarge, BatteryRoomTemp, 
    BatteryRoomHumid, BatteryAge, InvertEfficiency)
    Buildings[address] = building

#NYISO DATA
#load
loaddf = NYISOData(dataset='load_h', year='2023').df # year argument in local time, but returns dataset in UTC
otherloaddf =  NYISOData(dataset='load_h', year='2024').df # year argument in local time, but returns dataset in UTC
loaddf = pd.concat([loaddf, otherloaddf], ignore_index=False)
loaddf = loaddf.loc[:, ['N.Y.C.']]
loaddf = loaddf.reset_index()
loaddf.columns = ['Time', 'N.Y.C.']
loaddf['Time'] = loaddf['Time'] - pd.Timedelta(hours=5)
#price
priceedf = NYISOData(dataset='lbmp_dam_h', year='2024').df
priceedff = NYISOData(dataset='lbmp_dam_h', year='2023').df
priceedf = pd.concat([priceedf, priceedff], ignore_index=False)
priceedf = priceedf.loc[:, [('LBMP ($/MWHr)', 'N.Y.C.')]]
priceedf = priceedf.reset_index()
priceedf.columns = ['Time', 'N.Y.C.']
priceedf['Time'] = priceedf['Time'] - pd.Timedelta(hours=5)

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

priceedf['Time'] = pd.to_datetime(priceedf['Time'])
priceedf['Hour'] = priceedf['Time'].dt.hour
priceedf['saved_hour'] = priceedf['Hour']
priceedf = priceedf[priceedf['Time'].dt.date.isin(originalweatherdf['datetime'])]
priceedf['Time'] = priceedf.apply(
    lambda row: pd.to_datetime(row['Time'].date()) + pd.DateOffset(hours=row['saved_hour']), axis=1
)
priceedf = priceedf.drop(columns=['Hour', 'saved_hour'])
priceedf.set_index('Time', inplace=True)

#set hourly and daily load df
loadhourly = loaddf.copy()
loaddaily = loaddf.copy()

#set date as the index
loaddf.set_index('Time', inplace=True)
givenload = loaddf.copy()

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
Hspringdf = pd.read_csv(Springpath)
Hsummerdf = pd.read_csv(Summerpath)
Hwinterdf = pd.read_csv(Winterpath)
Hfalldf = pd.read_csv(FallPath)
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
print('Loop Started')
loopstart = time.time()
#Main Project
Blevel = 0 #represented as a percentage(.1 = 10%)
SUsed = 0
GUsed = 0
iterations=0
BasePanel = SolarPanel(500, .21, .05, .003, .995)
hourly_data_df['date'] = pd.to_datetime(hourly_data_df['date'])
originalweatherdf['datetime'] = pd.to_datetime(originalweatherdf['datetime'])
originalweatherdf.set_index('datetime', inplace=True)
#fix hourly_data_df
hourly_data_df['date'] = hourly_data_df.apply(lambda row: row['date'].replace(hour=row['hour']), axis=1)

# Initialize empty DataFrames for old and new loads
old_loads_df = pd.DataFrame(index=hourly_data_df['date'], columns=Buildings.keys())
new_loads_df = pd.DataFrame(index=hourly_data_df['date'], columns=Buildings.keys())

# Start main loop
for index, row in hourly_data_df.iterrows():
  date = row['date']
  hourly_solar = row['solarenergy']
  energyproduced = calcsolarpperiod(hourly_solar, BasePanel, originalweatherdf, date) #take into account solar panel efficiencies
  energyproduced *= BasePanel.ChargeControllerEfficiency
  for address in Buildings:
      miniloopstart = time.time()
      energy_stored = 0
      building = Buildings[address]
      energy_needed = (get_hourly_usage(date.hour,building.units * building.UsagePproperty))
      # Update old and new load DataFrames
      old_loads_df.loc[date, address] = (get_hourly_usage(date.hour,building.units * building.UsagePproperty))
      new_loads_df.loc[date, address] = (get_hourly_usage(date.hour,building.units * building.UsagePproperty))
      energy_obtained = building.TheoSpace * energyproduced

      #calc max charge and discharge rate, for one battery
      max_charge_rate = calcbatteryrates(building)
      max_discharge_rate = calcbatteryrates(building, True)

      #calc total charge rate for all batteries in building
      max_charge_rate *= building.numberofbatteries
      max_discharge_rate *= building.numberofbatteries

      # Prioritize battery storage with charging limit
      if building.BatteryLevel < 1:
          remainingstorage = (1 - building.BatteryLevel) * building.storagePbattery * building.numberofbatteries
          max_charge = min(remainingstorage, max_charge_rate)
          if energy_obtained >= max_charge:
              energy_stored = max_charge
              building.BatteryLevel +=  max_charge / (building.storagePbattery * building.numberofbatteries)
              energy_obtained -= max_charge
          else:
              energy_stored = energy_obtained
              building.BatteryLevel += energy_stored / (building.storagePbattery * building.numberofbatteries)
              energy_obtained -= energy_stored

      # Use stored energy if solar energy is insufficient
      energy_obtained *= building.InvertEfficiency
      if energy_obtained < energy_needed:
          energy_needed -= energy_obtained
          if building.BatteryLevel > 0:
                available_energy = building.BatteryLevel * building.storagePbattery * building.numberofbatteries
                if available_energy >= energy_needed:
                    discharge_amount = min(energy_needed, max_discharge_rate)
                    building.BatteryLevel -= discharge_amount / (building.storagePbattery * building.numberofbatteries)
                    SUsed += discharge_amount
                    # Update new load DataFrames
                    new_loads_df.loc[date, address] -= discharge_amount
                    loaddf.at[date, 'N.Y.C.'] -= discharge_amount
                    energy_needed -= discharge_amount
                else:
                    discharge_amount = min(available_energy, max_discharge_rate)
                    SUsed += discharge_amount
                    # Update new load DataFrames
                    new_loads_df.loc[date, address] -= discharge_amount
                    loaddf.at[date, 'N.Y.C.'] -= discharge_amount
                    energy_needed -= discharge_amount
                    building.BatteryLevel -= discharge_amount / (building.storagePbattery * building.numberofbatteries)
                    if building.BatteryLevel < 0:
                        building.BatteryLevel = 0
          GUsed += energy_needed
      else:
          SUsed += energy_needed
          new_loads_df.loc[date, address] -= energy_needed
          loaddf.at[date, 'N.Y.C.'] -= energy_needed
  miniloopend = time.time()
  print(f'miniloop time: {miniloopend-miniloopstart}')
  
  iterations +=1

#check how long loop takes to run
looptime = time.time()
looptime -= loopstart
print(f'Loop done in : {looptime} seconds!')

#calculate new ISO price data assuming Solar load decreases
loaddfcopy = loaddf.copy()
NewLBMPprices = pricingmodelcalc(loaddfcopy)
LBMPsavings = lbmpsavingscalc(NewLBMPprices)

#calculations for print statements
usageongrid = 0
for x in Buildings:
  usageongrid += (Buildings[x].units * Buildings[x].UsagePproperty)/24 # usagephour
expected_total_energy_used = iterations * usageongrid  # number of hours iterated * usage on whole grid per hour
TLoad = loaddf['N.Y.C.'].sum()

#Building Level Savings calculations
difference_in_loads = old_loads_df - new_loads_df
avgloadsavperdayperbuild = difference_in_loads.values.mean()

old_loads_df.index = old_loads_df.index.tz_localize(None)
new_loads_df.index = new_loads_df.index.tz_localize(None)
priceedf.index = priceedf.index.tz_localize(None)
old_total_price_df = old_loads_df.multiply(priceedf['N.Y.C.'], axis=0)
new_total_price_df = new_loads_df.multiply(priceedf['N.Y.C.'], axis=0)
old_total_price_df *= .000001
new_total_price_df *= .000001
difference_in_prices = old_total_price_df - new_total_price_df
avgpricesaveperdayperbuild = difference_in_prices.values.mean() 
print(old_total_price_df)
print(new_total_price_df)

end_time = time.time()
duration = end_time - start_time

#Use GUI
whattoprint = [(f'Total Hourly Iterations:{iterations}'),
(f'Total Solar Energy Used:       {SUsed} Wh'),
(f'Total Grid Energy Used:        {GUsed} Wh'),
(f'Total Energy Used:             {SUsed+GUsed} Wh'),
(f'Expected Total Energy Used:    {expected_total_energy_used} Wh'),
(f'Total load in NYC befor solar: {TloadBefore} wh'),
(f'Total load in NYC after Solar: {TLoad} wh'),
(f'Total LBMP price saved for the year ($/Mwh): ${LBMPsavings}'),
(f'Average price saved per hour: ${LBMPsavings/iterations}'),
(f'Net Savings after decreased load: ${netsavingscalc(lbmpsavingscalc(NewLBMPprices, True), loaddfcopy, givenload)}'),
(f'Average load saved per building per day: {avgloadsavperdayperbuild}'),
(f'Average net savings per building per day with solar: ${avgpricesaveperdayperbuild}'),
(f"Execution time: {duration} seconds")]
show_GUI(whattoprint)


#price p building, eui change, selling back to grid, if grid was all connected