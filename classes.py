import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from nyisotoolkit import NYISOData, NYISOStat, NYISOVis
import time
import tkinter as tk
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

class SolarPanel:
  def __init__(self, RatedPowerTotal,  Efficiency, Tolerance, TempCoefficient,ChargeControllerEfficiency, refrencetemp = 25, RatedPPC = 0):
    self.RatedPowerTotal = RatedPowerTotal
    self.RatedPPC = RatedPPC
    self.Efficiency = Efficiency
    self.Tolerance = Tolerance
    self.ChargeControllerEfficiency = ChargeControllerEfficiency
    self.refrencetemp = refrencetemp
    self.TempCoefficient = TempCoefficient

class Building:
  def __init__(self, units, TheoSpace, TotalSquareFeet, UsagePproperty,
              numberofbatteries, storagePbattery, MaxCharge, MaxDischarge,
              BatteryRoomTemp, BatteryRoomHumid, BatteryAge, InvertEfficiency,
              RetrofitPercentChange, AppliancePercentChange,
              MaterialPercentChange, BatteryLevel=0): #add charging factors
    self.units = units
    self.TheoSpace = TheoSpace
    self.TotalSquareFeet = TotalSquareFeet
    self.UsagePproperty = UsagePproperty
    self.numberofbatteries = numberofbatteries
    self.storagePbattery = storagePbattery
    self.MaxCharge = MaxCharge
    self.MaxDischarge = MaxDischarge
    self.BatteryRoomTemp = BatteryRoomTemp
    self.BatteryRoomHumid = BatteryRoomHumid
    self.BatteryAge = BatteryAge
    self.InvertEfficiency = InvertEfficiency
    self.RetrofitPercentChange = RetrofitPercentChange
    self.AppliancePercentChange = AppliancePercentChange
    self.MaterialPercentChange = MaterialPercentChange
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

def calcbatteryrates(buildingobject, chargeFalseordisTrue = False):
  def calculate_temp_effect(room_temp):
    #convert to C
    room_temp = (room_temp - 32) * (5/9)
    # Optimal temperature range for lithium-ion batteries
    optimal_temp_min = 20  # °C
    optimal_temp_max = 25  # °C
    if room_temp < optimal_temp_min:
        # Reduced efficiency due to cold temperatures
        return max(0.8, (room_temp - 10) / optimal_temp_min)  # Assuming linear degradation
    elif room_temp > optimal_temp_max:
        # Reduced efficiency due to high temperatures
        return max(0.8, (40 - room_temp) / (40 - optimal_temp_max))  # Assuming linear degradation
    else:
        # Optimal efficiency
        return 1.0
  def calculate_humidity_effect(room_humidity):
    room_humidity *= 100
    # Optimal humidity range for battery rooms
    optimal_humidity_min = 40  # %
    optimal_humidity_max = 60  # %
    if room_humidity < optimal_humidity_min:
        # Lower end of the optimal range; no significant impact
        return 1.0
    elif room_humidity > optimal_humidity_max:
        # Reduced efficiency due to high humidity
        return max(0.9, (80 - room_humidity) / (80 - optimal_humidity_max))  # Assuming linear degradation
    else:
        # Optimal efficiency
        return 1.0
  def calculate_battery_age_factor(age_years):
    # Define parameters for the degradation model
    max_degradation = 0.3     # Maximum degradation factor (e.g., 30% loss in efficiency)
    # Exponential decay model for degradation
    # This model assumes that degradation accelerates over time
    degradation_factor = 1 - max_degradation * (1 - np.exp(-0.05 * age_years))
    # Ensure that the factor is not less than 0 (though practical batteries would not degrade to 0%)
    return max(degradation_factor, 0.0)
  if chargeFalseordisTrue == False:
     return buildingobject.MaxCharge * calculate_temp_effect(buildingobject.BatteryRoomTemp) * calculate_humidity_effect(buildingobject.BatteryRoomHumid) * calculate_battery_age_factor(buildingobject.BatteryAge)
  else: 
     return buildingobject.MaxDischarge * calculate_temp_effect(buildingobject.BatteryRoomTemp) * calculate_humidity_effect(buildingobject.BatteryRoomHumid) * calculate_battery_age_factor(buildingobject.BatteryAge)

def get_hourly_usage(hour, total_daily_usage,building, IncludeRetrofit = False):
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

    if IncludeRetrofit == False:
       return hourly_usage
    
    #Account for Usage Savings
    else:
      hourly_usage *= (1-building.RetrofitPercentChange)
      hourly_usage *= (1-building.AppliancePercentChange)
      hourly_usage *= (1-building.MaterialPercentChange)
      return hourly_usage

def pricingmodelcalc(postsolarload):
  #configure ISO DATA
  llloaddf = NYISOData(dataset='load_h', year='2024').df
  lloaddf = NYISOData(dataset='load_h', year='2023').df # year argument in local time, but returns dataset in UTC
  otherlloaddf =  NYISOData(dataset='load_h', year='2022').df # year argument in local time, but returns dataset in UTC
  otherlloaddff =  NYISOData(dataset='load_h', year='2021').df
  otherlloaddfff =  NYISOData(dataset='load_h', year='2020').df
  otherlloaddffff =  NYISOData(dataset='load_h', year='2019').df
  lloaddf = pd.concat([llloaddf, lloaddf], ignore_index=False)
  lloaddf = lloaddf.loc[:, ['N.Y.C.']]
  lloaddf = lloaddf.reset_index()
  lloaddf.columns = ['Time', 'N.Y.C.']
  
  lloaddf['Time'] = pd.to_datetime(lloaddf['Time'])
  priced = NYISOData(dataset='lbmp_dam_h', year='2024').df
  pricedf = NYISOData(dataset='lbmp_dam_h', year='2023').df
  pricedff = NYISOData(dataset='lbmp_dam_h', year='2022').df
  pricedfff= NYISOData(dataset='lbmp_dam_h', year='2021').df
  pricedffff = NYISOData(dataset='lbmp_dam_h', year='2020').df
  pricedfffff = NYISOData(dataset='lbmp_dam_h', year='2019').df
  pricedf = pd.concat([priced, pricedf], ignore_index=False)
  pricedf = pricedf.loc[:, [('LBMP ($/MWHr)', 'N.Y.C.')]]
  pricedf = pricedf.reset_index()
  pricedf.columns = ['Time', 'N.Y.C.']
  lloaddf['N.Y.C.'] = lloaddf['N.Y.C.'] * 1000000 #convert MW to W
  pricedf['Time'] = pd.to_datetime(pricedf['Time'])
  lloaddf['Time'] = lloaddf['Time'] - pd.Timedelta(hours=5)
  pricedf['Time'] = pricedf['Time'] - pd.Timedelta(hours=5)

  #configure given load df
  postsolarload.reset_index(inplace=True)
  postsolarload['Time'] = pd.to_datetime(postsolarload['Time'])
  postsolarload['day_of_week'] = postsolarload['Time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
  postsolarload['month'] = postsolarload['Time'].dt.month
  postsolarload['hour'] = postsolarload['Time'].dt.hour
  postsolarload.rename(columns={'N.Y.C.': 'N.Y.C._x'}, inplace=True)

  #merge data
  data = pd.merge(lloaddf, pricedf, on='Time')
  data['day_of_week'] = data['Time'].dt.dayofweek  # 0 = Monday, 6 = Sunday
  data['month'] = data['Time'].dt.month
  data['hour'] = data['Time'].dt.hour

  #Random Forest Model
  features = ['N.Y.C._x', 'day_of_week', 'month', 'hour']
  X = data[features]
  y = data['N.Y.C._y']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=32)

  # Model tuning with GridSearchCV
  #param_grid = {
  #    'n_estimators': [100, 200],
  #    'max_depth': [None, 10, 20],
  #    'min_samples_split': [2, 5],
  #    'min_samples_leaf': [1, 4]
  #}
    
  #model = GridSearchCV(RandomForestRegressor(random_state=32), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
  model = RandomForestRegressor(n_estimators=100, random_state=32)
  model.fit(X_train, y_train)
    
  # Best model
  #best_model = model.best_estimator_
    
  # Cross-validation
  #cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
  #print(f'Cross-Validation Mean Squared Error: {-cv_scores.mean()}')
  predictions = model.predict(X_test)
  # Create scatter plot with y=x line
  fig, ax = plt.subplots()
  ax.scatter(y_test, predictions)
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # y=x line
  ax.set_xlabel('Actual Prices')
  ax.set_ylabel('Predicted Prices')
  ax.set_title('Actual vs Predicted Prices')
  plt.show(block=False)
  mae = mean_absolute_error(y_test, predictions)
  mse = mean_squared_error(y_test, predictions)
  rmse = mse ** .5
  print(f'Mean Absolute Error: {mae}')
  print(f'Mean Squared Error: {mse}')
  print(f'Root Mean Squared Error: {rmse}')

  #use model
  new_prices = model.predict(postsolarload[['N.Y.C._x', 'day_of_week', 'month', 'hour']])
  postsolarload['predicted_price'] = new_prices
  postsolarload = postsolarload.drop(columns=['day_of_week', 'month', 'hour'])
  return postsolarload

def lbmpsavingscalc(newprices, dfIFNOTvalue = False, dropgivenandnew = False):
  ppricedf = NYISOData(dataset='lbmp_dam_h', year='2023').df
  ppricedff = NYISOData(dataset='lbmp_dam_h', year='2024').df
  ppricedf = pd.concat([ppricedf, ppricedff], ignore_index=False)
  ppricedf = ppricedf.loc[:, [('LBMP ($/MWHr)', 'N.Y.C.')]]
  ppricedf = ppricedf.reset_index()
  ppricedf.columns = ['Time', 'N.Y.C.']
  ppricedf['Time'] = pd.to_datetime(ppricedf['Time'])
  ppricedf['Time'] = ppricedf['Time'] - pd.Timedelta(hours=5)
  new_dates = newprices['Time'].dt.date.unique()
  filtered_given = ppricedf[ppricedf['Time'].dt.date.isin(new_dates)]
  merged = filtered_given
  merged['N.Y.C._new'] = newprices['predicted_price']
  merged.columns = ['Time','N.Y.C._given', 'N.Y.C._new']
  merged['price_difference'] = merged['N.Y.C._given'] - merged['N.Y.C._new']
  total_savings = merged['price_difference'].sum()
  if dfIFNOTvalue == False:
    return merged['price_difference'].sum()
  else:
    if dropgivenandnew == False:
      return merged
    else:
      merged = merged.drop(columns=['N.Y.C._new', 'N.Y.C._given'])
      return merged
  
def netsavingscalc(mergedpricedf, postsolarload, givenloaddf, dfIFNOTvalue = False):
  mergedpricedf['giventotalpricephour'] = mergedpricedf['N.Y.C._given'] * (givenloaddf['N.Y.C.']/1000000)
  mergedpricedf['newtotalpricephour'] = mergedpricedf['N.Y.C._new'] * (postsolarload['N.Y.C._x']/1000000)
  if dfIFNOTvalue == True:
    return mergedpricedf['giventotalpricephour'] - mergedpricedf['newtotalpricephour'] 
  return mergedpricedf['giventotalpricephour'].sum() - mergedpricedf['newtotalpricephour'].sum()

def AnnualEUIsavCalc(old_load, BuildingsDictionary):
  new_lis = []
  for address in BuildingsDictionary:
     avgloadphour = old_load[address].mean()
     EUIsavforbuild = (avgloadphour*24*365*.001)/BuildingsDictionary[address].TotalSquareFeet
     new_lis.append(EUIsavforbuild)
  avgEUIsav = sum(new_lis)/len(new_lis)
  return avgEUIsav
  

def show_GUI(whattoprint):
    # Create the GUI window
    root = tk.Tk()
    root.title("Results")
    # Display the results in the window
    text = tk.Text(root, wrap="word", width=100, height=20)
    text.pack(expand=True, fill="both")
    # Insert the results into the text widget
    for printstatement in whattoprint:
        text.insert(tk.END, printstatement + "\n")
    # Start the GUI event loop
    root.mainloop()