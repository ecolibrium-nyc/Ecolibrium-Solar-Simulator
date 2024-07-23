import random
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