# Running in Xavier
# Wrote by Yuan He in Beijing Normal University
# 2021.10.10

from keras import layers
from keras import losses
from keras import models
from keras import metrics
from keras import optimizers
from keras import callbacks
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
np.random.seed(0)

# Load model
model_DRGN = models.load_model('./DRGN_SDD_Landsat8')

# Load training and validating data
train_x1 = np.load('./input_real/l8_train_x.npy')
valid_x1 = np.load('./input_real/l8_valid_x.npy')
train_y1 = np.load('./input_real/l8_train_Y.npy')
valid_y1 = np.load('./input_real/l8_valid_Y.npy')

# Normalization setup
input_df = pd.read_csv('./landsat8.csv')
input = input_df[['B1','B2','B3','B4','B5','B7']].values
scaler_input = MinMaxScaler()
scaler_input.fit(input)

# Calculate Time Series
dft = pd.read_csv('./GEE_out/global_10more_time-v6-year.csv')
dft.pop('system:index')
dft.pop('.geo')
dft = dft.sort_values(by=['Long', 'Year'], ascending=[True, True])
print(dft)

input_col = ['B1','B2','B3','B4','B5','B7']
input = dft[input_col].values
print(input.shape)

pred_lake_dgrn = np.zeros(shape=input.shape[0])
for i in range(input.shape[0]):
	if (input[i,0] > 0) and (input[i,1] > 0) and (input[i,2] > 0) and (input[i,3] > 0):
		input_data = scaler_input.transform(input[i,:].reshape(1,-1)*10000).reshape(1,1,6)
		pred_lake_dgrn[i] = np.squeeze(model_DRGN.predict(input_data))
	else:
		pred_lake_dgrn[i] = np.nan
del i,input_data
print('The number of lake:'+str(pred_lake_dgrn.shape[0]/7))
dft['SDD_DGRN'] = list(pred_lake_dgrn.reshape(-1))
dft.to_csv('./GEE_out/Predict_global_SDD.csv')

'''Spatial pattern'''
# Acquire SDD from each year
dft2 = dft.sort_values(by=['Lat', 'Long'], ascending=[True, True])
print(dft2)
df_time = pd.DataFrame()
# Comput 2014
pred_lake_dgrn_2014 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2014)
lake_lat = np.zeros_like(pred_lake_dgrn_2014)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2014[lake] = dft['SDD_DGRN'].values[lake*7+0]
  lake_lon[lake] = dft['Long'].values[lake*7+0]
  lake_lat[lake] = dft['Lat'].values[lake*7+0]
del lake
df_time['Long'] = lake_lon
df_time['Lat'] = lake_lat
df_time['2014'] = pred_lake_dgrn_2014
del pred_lake_dgrn_2014,lake_lon,lake_lat

# Comput 2015
pred_lake_dgrn_2015 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2015)
lake_lat = np.zeros_like(pred_lake_dgrn_2015)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2015[lake] = dft['SDD_DGRN'].values[lake*7+1]
  lake_lon[lake] = dft['Long'].values[lake*7+1]
  lake_lat[lake] = dft['Lat'].values[lake*7+1]
del lake
# df_time['Long2'] = lake_lon
# df_time['Lat2'] = lake_lat
df_time['2015'] = pred_lake_dgrn_2015
del pred_lake_dgrn_2015,lake_lon,lake_lat

# Comput 2016
pred_lake_dgrn_2016 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2016)
lake_lat = np.zeros_like(pred_lake_dgrn_2016)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2016[lake] = dft['SDD_DGRN'].values[lake*7+2]
  lake_lon[lake] = dft['Long'].values[lake*7+2]
  lake_lat[lake] = dft['Lat'].values[lake*7+2]
del lake
# df_time['Long3'] = lake_lon
# df_time['Lat3'] = lake_lat
df_time['2016'] = pred_lake_dgrn_2016
del pred_lake_dgrn_2016,lake_lon,lake_lat

# Comput 2017
pred_lake_dgrn_2017 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2017)
lake_lat = np.zeros_like(pred_lake_dgrn_2017)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2017[lake] = dft['SDD_DGRN'].values[lake*7+3]
  lake_lon[lake] = dft['Long'].values[lake*7+3]
  lake_lat[lake] = dft['Lat'].values[lake*7+3]
del lake
# df_time['Long4'] = lake_lon
# df_time['Lat4'] = lake_lat
df_time['2017'] = pred_lake_dgrn_2017
del pred_lake_dgrn_2017,lake_lon,lake_lat

# Comput 2018
pred_lake_dgrn_2018 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2018)
lake_lat = np.zeros_like(pred_lake_dgrn_2018)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2018[lake] = dft['SDD_DGRN'].values[lake*7+4]
  lake_lon[lake] = dft['Long'].values[lake*7+4]
  lake_lat[lake] = dft['Lat'].values[lake*7+4]
del lake
# df_time['Long5'] = lake_lon
# df_time['Lat5'] = lake_lat
df_time['2018'] = pred_lake_dgrn_2018
del pred_lake_dgrn_2018,lake_lon,lake_lat

# Comput 2019
pred_lake_dgrn_2019 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2019)
lake_lat = np.zeros_like(pred_lake_dgrn_2019)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2019[lake] = dft['SDD_DGRN'].values[lake*7+5]
  lake_lon[lake] = dft['Long'].values[lake*7+5]
  lake_lat[lake] = dft['Lat'].values[lake*7+5]
del lake
# df_time['Long6'] = lake_lon
# df_time['Lat6'] = lake_lat
df_time['2019'] = pred_lake_dgrn_2019
del pred_lake_dgrn_2019,lake_lon,lake_lat

# Comput 2020
pred_lake_dgrn_2020 = np.zeros(shape=(int(pred_lake_dgrn.shape[0]/7),))
lake_lon = np.zeros_like(pred_lake_dgrn_2020)
lake_lat = np.zeros_like(pred_lake_dgrn_2020)
for lake in range(int(pred_lake_dgrn.shape[0]/7)):
  pred_lake_dgrn_2020[lake] = dft['SDD_DGRN'].values[lake*7+6]
  lake_lon[lake] = dft['Long'].values[lake*7+6]
  lake_lat[lake] = dft['Lat'].values[lake*7+6]
del lake
# df_time['Long7'] = lake_lon
# df_time['Lat7'] = lake_lat
df_time['2020'] = pred_lake_dgrn_2020
del pred_lake_dgrn_2020,lake_lon,lake_lat

# Put temporal trend (2014-2020) out 
df_time.to_csv('./GEE_out/2014_to_2020_SDD_DGRN.csv')

'''Temporal trends'''
trend = ['2014','2015','2016','2017','2018','2019','2020']
sdd_time = np.array(df_time[trend].values)
print(sdd_time.shape)

from scipy.stats.mstats import theilslopes
from scipy.stats import kendalltau, linregress

# Time scale
x = np.arange(2014,2021)

# Calculate Sen's slope and pvalue
sen_slope = np.zeros(shape=(sdd_time.shape[0],))
mk_pvalue = np.zeros_like(sen_slope)
lin_slope = np.zeros_like(sen_slope)
lr_pvalue = np.zeros_like(sen_slope)
for t in range(sdd_time.shape[0]):
  # Mann-Kendall Trend Analysis
  sen_slope[t] = theilslopes(sdd_time[t,:], x, alpha=0.95)[0]
  mk_pvalue[t] = kendalltau(x, sdd_time[t,:])[1]
  # Linear Regresson
  lin_slope[t] = linregress(x, sdd_time[t,:])[0]
  lr_pvalue[t] = linregress(x, sdd_time[t,:])[3]
del t

# Put slope and pvalue out to .csv
df_time['Sen_Slope'] = sen_slope
df_time['MK_pvalue'] = mk_pvalue
df_time['Linear_Slope'] = lin_slope
df_time['LR_pvalue'] = lr_pvalue
df_time.to_csv('./GEE_out/Trends_2014_to_2020_SDD_DGRN.csv')

# Delete Greenland
Greenland_loc = []
for t in df_time.index:
  if df_time['Long'].values[t] > -60 and df_time['Long'].values[t] < -10:
    if df_time['Lat'].values[t] > 58 and df_time['Lat'].values[t] < 85:
      Greenland_loc.append(t)
  elif df_time['Long'].values[t] > -100 and df_time['Long'].values[t] < -10:
    if df_time['Lat'].values[t] > 75.2 and df_time['Lat'].values[t] < 85:
      Greenland_loc.append(t)
df_noGreenland = df_time.drop(Greenland_loc)
df_noGreenland.to_csv('./GEE_out/Trends_2014_to_2020_SDD_DGRN_no_Greenland.csv')
