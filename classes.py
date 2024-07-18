class SolarPanel:
  def __init__(self, RatedPowerTotal,  Efficiency, Tolerance, TempCoefficient,RatedPPC = 0):
    self.RatedPowerTotal = RatedPowerTotal
    self.RatedPPC = RatedPPC
    self.Efficiency = Efficiency
    self.Tolerance = Tolerance
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

#def calcsolarpday(solarenergy, solarpanel):
  