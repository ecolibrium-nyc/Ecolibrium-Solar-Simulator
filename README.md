This is a repository of a program that will simulate various scenarios and will predict the gain that would result in using solar panels on different buildings in the Lower East Side of Manhattan.

This code is transferable to any location as long as CSV files are interchanged appropriately.

The weather data was sourced from Visual Crossing. Information on the raw data source can be found here: https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-sources-and-attribution/

The building data is currently randomized with approximate estimations for 30 buildings. When implemented, the building data will be sourced from Loisaida Inc. 

Overarching Goals: 
1. Simulate NYC load demand if solar panels where incorporated using NYISO historic load data.
2. Simulate electricity price using a regression model through past demand and price data from the NYISO. 
3. Simulate charging and transmission capactities and their effect on the grid and price.
4. Incorporate the increased resilience from the decreased loads, savings calculated from the regression model, and enviornmental impact into a cost-benefit analysis.


PLANS: FIX LOAD BUG, MULTIPROCESS THE INNER FOR LOOP, FIX DICT PROBLEM