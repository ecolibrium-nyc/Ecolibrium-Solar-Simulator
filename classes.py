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

def pricingmodelcalc(postsolarload):
  #configure ISO DATA
  lloaddf = NYISOData(dataset='load_h', year='2023').df # year argument in local time, but returns dataset in UTC
  otherlloaddf =  NYISOData(dataset='load_h', year='2022').df # year argument in local time, but returns dataset in UTC
  otherlloaddff =  NYISOData(dataset='load_h', year='2021').df
  otherlloaddfff =  NYISOData(dataset='load_h', year='2020').df
  otherlloaddffff =  NYISOData(dataset='load_h', year='2019').df
  lloaddf = pd.concat([lloaddf, otherlloaddf, otherlloaddff, otherlloaddfff, otherlloaddffff], ignore_index=False)
  lloaddf = lloaddf.loc[:, ['N.Y.C.']]
  lloaddf = lloaddf.reset_index()
  lloaddf.columns = ['Time', 'N.Y.C.']
  
  lloaddf['Time'] = pd.to_datetime(lloaddf['Time'])
  pricedf = NYISOData(dataset='lbmp_dam_h', year='2023').df
  pricedff = NYISOData(dataset='lbmp_dam_h', year='2022').df
  pricedfff= NYISOData(dataset='lbmp_dam_h', year='2021').df
  pricedffff = NYISOData(dataset='lbmp_dam_h', year='2020').df
  pricedfffff = NYISOData(dataset='lbmp_dam_h', year='2019').df
  pricedf = pd.concat([pricedf, pricedff, pricedfff, pricedffff, pricedfffff], ignore_index=False)
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
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
  model = RandomForestRegressor(n_estimators=500, random_state=42)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  plt.scatter(y_test, predictions)
  plt.xlabel('Actual Prices')
  plt.ylabel('Predicted Prices')
  plt.title('Actual vs Predicted Prices')
  #plt.show()
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
  print(f"Total Price Savings: {total_savings}")
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

def show_GUI(whattoprint):
    # Create the GUI window
    root = tk.Tk()
    root.title("Results")
    # Display the results in the window
    text = tk.Text(root, wrap="word", width=60, height=20)
    text.pack(expand=True, fill="both")
    # Insert the results into the text widget
    for printstatement in whattoprint:
        text.insert(tk.END, printstatement + "\n")
    # Start the GUI event loop
    root.mainloop()