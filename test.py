from nyisotoolkit import NYISOData, NYISOStat, NYISOVis
df = NYISOData(dataset='load_h', year='2022').df
df.to_csv(r'C:\Users\danie\Downloads\Loisaida Code\2022load.csv')