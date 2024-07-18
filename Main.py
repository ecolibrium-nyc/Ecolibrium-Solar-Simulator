import pandas
import math
from nyisotoolkit import NYISOData, NYISOStat, NYISOVis
from classes import *

#Weather Data
originalweatherdf = pandas.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\10002 2023-07-18 to 2024-07-16.csv') #read weather csv file
originalweatherdf['sunrise'] = pandas.to_datetime(originalweatherdf['sunrise']) #cnvert to datetime format
originalweatherdf['sunset'] = pandas.to_datetime(originalweatherdf['sunset'])
originalweatherdf['day_length'] = originalweatherdf['sunset'] - originalweatherdf['sunrise'] #calc day length
originalweatherdf = originalweatherdf.drop(columns=['name', 'description', 'stations', 'sunrise', 'sunset']) #remove columns

originalweatherdf['solarenergy'] = originalweatherdf['solarenergy'] * 227.78 #convert MJ/m^2 to wh/m^2
originalweatherdf = originalweatherdf[originalweatherdf['solarenergy'] != 0] #remove all days w/ unmeasured solar energies(21)
print(originalweatherdf)

AVGLD = originalweatherdf['day_length'].mean()
AVGSE = originalweatherdf['solarenergy'].mean()

print('AVG Solar Energy (wh/m^2) for the year ' + str(AVGSE))
print('AVG Day Length for the year ' + str(AVGLD))

#Building Data

buildingdf = pandas.read_csv(r'C:\Users\danie\Downloads\Loisaida Code\Building Data.csv') #read Building csv file
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
loaddf = pandas.concat([loaddf, otherloaddf], ignore_index=False)
loaddf = loaddf.loc[:, ['N.Y.C.']]
loaddf = loaddf.reset_index()
loaddf.columns = ['Time', 'N.Y.C.']

#Match Weather Data to NYISO Data

originalweatherdf['datetime'] = pandas.to_datetime(originalweatherdf['datetime']).dt.date
loaddf['Time'] = pandas.to_datetime(loaddf['Time']).dt.date
loaddf = loaddf[loaddf['Time'].isin(originalweatherdf['datetime'])]
loaddf['N.Y.C.'] = loaddf['N.Y.C.'] * 1000000 #convert MW to W

#set hourly and daily load df
loadhourly = loaddf.copy()
loaddaily = loaddf.copy()

#Convert to daily structure
loaddaily['Time'] = pandas.to_datetime(loaddaily['Time'])
loaddaily['Time'] = loaddaily['Time'].dt.date
loaddaily = loaddaily.groupby('Time')['N.Y.C.'].sum().reset_index()
print(loaddaily)
TloadBefore = loaddaily['N.Y.C.'].sum()


#Weather DF
originalweatherdf['WeatherClass'] = Weather(originalweatherdf['day_length'], originalweatherdf['preciptype'], originalweatherdf['temp'])
Weatherdf = originalweatherdf['WeatherClass']
originalweatherdf = originalweatherdf.drop(columns=['WeatherClass'])

#Main Project
Blevel = 0 #represented as a percentage(.1 = 10%)
SUsed = 0
GUsed = 0
iterations=0
"""Buildings = {'Address1': Building(100, 2500, 10000, 200, 5000),
             'Address2': Building(50, 2500, 10000, 100, 5000),
             'Address3': Building(200, 2500, 10000, 400, 5000),
             'Address4': Building(1000, 2500, 10000, 2000, 5000)}"""
BasePanel = SolarPanel(500, .2, 1, 1, 1)
powerpsqm = originalweatherdf['solarenergy'] * BasePanel.Efficiency #calc total solar input per sq m
for daily_solar in powerpsqm:

  for address in Buildings:
      energy_stored = 0
      energy_needed = Buildings[address].units * Buildings[address].UsagePproperty
      energy_obtained = Buildings[address].TheoSpace * daily_solar

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
                  loaddaily.at[iterations, 'N.Y.C.'] -= energy_needed
                  energy_needed = 0
              else:
                  SUsed += Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  loaddaily.at[iterations, 'N.Y.C.'] -= Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  energy_needed -= Buildings[address].BatteryLevel * Buildings[address].storagePbattery * Buildings[address].numberofbatteries
                  Buildings[address].BatteryLevel = 0
          GUsed += energy_needed
      else:
          SUsed += energy_needed
          loaddaily.at[iterations, 'N.Y.C.'] -= energy_needed
  iterations +=1

print(iterations)
print(f'Total Solar Energy Used:       {SUsed} Wh')
print(f'Total Grid Energy Used:        {GUsed} Wh')
print(f'Total Energy Used:             {SUsed+GUsed} Wh')
usageongrid = 0
for x in Buildings:
  usageongrid += Buildings[x].units * Buildings[x].UsagePproperty
expected_total_energy_used = 344 * usageongrid  # 344 days * usage on whole grid per day
print(f'Expected Total Energy Used:    {expected_total_energy_used} Wh')
TLoad = loaddaily['N.Y.C.'].sum()
print(f'Total load in NYC befor solar: {TloadBefore} wh')
print(f'Total load in NYC after Solar: {TLoad} wh')