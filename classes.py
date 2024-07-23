import random
import numpy as np
import pandas as pd

class SolarPanel:
  def __init__(self, RatedPowerTotal,  Efficiency, Tolerance, TempCoefficient,refrencetemp = 25, RatedPPC = 0):
    self.RatedPowerTotal = RatedPowerTotal
    self.RatedPPC = RatedPPC
    self.Efficiency = Efficiency
    self.Tolerance = Tolerance
    self.refrencetemp = refrencetemp
    self.TempCoefficient = TempCoefficient

class Building:
  def __init__(self, units, TheoSpace, UsagePproperty, numberofbatteries, storagePbattery, BatteryLevel=0): #add charging factors
    self.units = units
    self.TheoSpace = TheoSpace
    self.UsagePproperty = UsagePproperty
    self.numberofbatteries = numberofbatteries
    self.storagePbattery = storagePbattery
    self.BatteryLevel = BatteryLevel
  def __str__(self):
    return f"Building(units={self.units}, TheoSpace={self.TheoSpace}, UsagePproperty={self.UsagePproperty}, numberofbatteries={self.numberofbatteries}, storagePbattery={self.storagePbattery}, BatteryLevel={self.BatteryLevel})"

class Weather:
  def __init__(self, SunDuration, Precipitation, temp):
    self.SunDuration = SunDuration
    self.Precipitation = Precipitation
    self.temp = temp
  def __str__(self):
    return "The Duration of the sun is " + str(self.SunDuration) + " with an average temp of " + str(self.temp) + " and " + str(self.Precipitation) + " Precipitation type!"

def calcsolarpperiod(solarenergy, solarpanel, weatherdf, dates):
  solarforday = solarenergy * solarpanel.Efficiency #takes into account efficiency
  solarforday *= (1 + round(random.uniform(0, solarpanel.Tolerance), 5)) #takes into account tolerance rating on a random scale up to 5 decimal points
  solarforday *= (1-solarpanel.TempCoefficient*(weatherdf.loc[dates.strftime('%Y-%m-%d'), 'temp'] - solarpanel.refrencetemp)) #take into account weather
  return solarforday

def get_hourly_usage(hour, total_daily_usage):
    def usage_pattern(hour):
      if 6 <= hour < 9:
        return 0.2 * (hour - 6) / 3
      elif 9 <= hour < 15:
        return 0.1
      elif 15 <= hour < 21:
        return 0.3 - 0.01 * (hour - 15)
      else:
        return 0.1

    hours = np.arange(24)
    pattern = np.array([usage_pattern(h) for h in hours])
    
    # Check for NaN values and replace with zero
    pattern = np.nan_to_num(pattern)

    # Check if the sum is zero before dividing
    pattern_sum = pattern.sum()
    if pattern_sum == 0:
        raise ValueError("The sum of the pattern is zero, cannot normalize")

    pattern /= pattern_sum

    # Get the usage proportion for the given hour
    usage_proportion = pattern[hour]
    
    # Calculate the hourly usage
    hourly_usage = usage_proportion * total_daily_usage
    return hourly_usage