import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import xlrd
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import callbacks, models
from scipy.interpolate import interpn

# define mape
def mape(y_true, y_pred):
  n = len(y_true)
  mape = sum(np.abs((y_true-np.squeeze(y_pred))/y_true))/n*100
  return mape

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)
# 8
train_x3 = np.load('./input_real/l8_train_x.npy')
valid_x3 = np.load('./input_real/l8_valid_x.npy')
train_y3 = np.load('./input_real/l8_train_Y.npy')
valid_y3 = np.load('./input_real/l8_valid_Y.npy')

df = pd.read_csv('landsat8.csv')
from sklearn.preprocessing import MinMaxScaler
input = df[['B1','B2','B3','B4','B5','B7']].values
scaler_input = MinMaxScaler()
scaler_input.fit(input)
input_nor = scaler_input.transform(train_x3)
valid_nor = scaler_input.transform(valid_x3)

model = Sequential()
model.add(GRU(512, input_shape=(None, input_nor.shape[-1]), return_sequences=True))
model.add(BatchNormalization())
model.add(GRU(512, return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())

model.add(GRU(512, return_sequences=True))
model.add(BatchNormalization())
model.add(GRU(512, return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')
print(model.summary())
cb = callbacks.EarlyStopping(monitor='loss', mode='min', patience=100)
history = model.fit(input_nor.reshape(input_nor.shape[0],1,input_nor.shape[-1]), train_y3, 
                    steps_per_epoch=10, epochs=5000, verbose=2,
                    callbacks=cb)

# model.save('DRGN_SDD_Landsat8')
model_DGRN = models.load_model('./DRGN_SDD_Landsat8')

pred_rnn = np.squeeze(model_DGRN.predict(valid_nor.reshape(valid_nor.shape[0],1,6)))
trad_rnn = np.squeeze(model_DGRN.predict(input_nor.reshape(input_nor.shape[0],1,6)))

print("R2 test:"+str(r2_score(valid_y3, pred_rnn)))

from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    if ax is None :
        fig , ax = plt.subplots(figsize=(7,6))
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density=True)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax1 = ax.scatter( x, y, c=z, **kwargs )
    ax.set_xticks([0,2,4,6,8,10,12,14,16],[0,2,4,6,8,10,12,14,16])
    ax.set_yticks([0,2,4,6,8,10,12,14,16],[0,2,4,6,8,10,12,14,16])
    ax.set_xlim([0,12])
    ax.set_ylim([0,12])
    ax.set_xlabel('Measured SDD',fontsize=18)
    ax.set_ylabel('Predicted SDD',fontsize=18)
    ax.plot(np.arange(-5,30), np.arange(-5,30), 'k','--')
    bar = fig.colorbar(ax1, ax=ax)
    bar.set_label('Density', labelpad=-25, y=1.06, rotation=0, fontsize=12)#, rotation=270)
    return ax1, ax,fig

# Train
ax1, ax, fig = density_scatter(train_y3, trad_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(train_y3, trad_rnn),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(train_y3, trad_rnn),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(train_y3, trad_rnn)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(train_y3, trad_rnn),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(a)DGRN: Training', fontsize=24)
plt.savefig('Fig3a.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(valid_y3, pred_rnn),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(valid_y3, pred_rnn),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(valid_y3, pred_rnn)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(valid_y3, pred_rnn),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(b)DGRN: Testing', fontsize=24)
plt.savefig('Fig3b.png', dpi=1000, bbox_inches='tight')
