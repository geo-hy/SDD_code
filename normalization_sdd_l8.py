# Environment set-up
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import xlrd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic, WhiteKernel, DotProduct
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import random as python_random
import sklearn.ensemble
from scipy.interpolate import interpn
import keras
from keras.layers import *
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import datetime
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(123)

'''Generate dataset'''
input_var = ['B1','B2','B3','B4','B5','B7']
output_var = ['SDD']
# landsat8
df2 = pd.read_csv('./landsat8.csv')
l8 = np.array(df2[input_var+output_var])
# # spilt
# l8_train_X, l8_valid_X, l8_train_Y, l8_valid_Y = train_test_split(l8[:,0:6], l8[:,6], test_size=0.2, random_state=0)

# save_all_data

# np.save('/content/drive/MyDrive/kd_sdd_data/input_real/l8_train_x.npy',l8_train_X)
# np.save('/content/drive/MyDrive/kd_sdd_data/input_real/l8_valid_x.npy',l8_valid_X)
# np.save('/content/drive/MyDrive/kd_sdd_data/input_real/l8_train_Y.npy',l8_train_Y)
# np.save('/content/drive/MyDrive/kd_sdd_data/input_real/l8_valid_Y.npy',l8_valid_Y)

# load_all_data
l8_train_X = np.load('./input_real/l8_train_x.npy')
l8_valid_X = np.load('./input_real/l8_valid_x.npy')
l8_train_Y = np.load('./input_real/l8_train_Y.npy')
l8_valid_Y = np.load('./input_real/l8_valid_Y.npy')

''' Normalization '''
# Normalization
# landsat8
scaler_x2 = MinMaxScaler()
scaler_x2.fit(l8[:,0:6])
x2train = scaler_x2.transform(l8_train_X)
x2valid = scaler_x2.transform(l8_valid_X)

# define mape
def mape(y_true, y_pred):
  n = len(y_true)
  mape = sum(np.abs((y_true-np.squeeze(y_pred))/y_true))/n*100
  return mape

''' MLPNN '''
# Multi-Layer Perceptron Neural Network Model
def getANN(train_x, train_y, n_layers, nns, batch=None, learning_rate=0.001):
  print('********** N_LAYERS='+str(n_layers)+' **********')
  print('##### N_NEURAL='+str(nns)+' #####')
  model = Sequential()
  model.add(Dense(nns,activation='relu', input_shape=(train_x.shape[1],), name='input'))
  for num in range(n_layers):
    model.add(Dense(nns,activation='relu'))
  model.add(Dense(1, name='output'))
  #optimizers = keras.optimizers.Adam(lr=learning_rate)
  model.compile(optimizer = 'adam', loss = 'mse')
#   print(model.summary())
  my_callbacks = [EarlyStopping(monitor='val_loss',patience=50,verbose=2,mode='auto')]
  # my_callbacks = [EarlyStopping(patience=10,verbose=2,mode='auto')]
  if batch == None:
    history = model.fit(train_x, train_y,epochs=5000,verbose=0,validation_split=0.1) #,batch_size=8, callbacks=my_callbacks, validation_split=0.1
  else:
    history = model.fit(train_x, train_y,epochs=5000,verbose=0,batch_size=batch,validation_split=0.1)
  return model, history

# landsat8
model2, history = getANN(x2train, l8_train_Y, 2, 8, learning_rate=0.001)
pred_ann2 = np.squeeze(model2.predict(x2valid))
pred_ann2_train = np.squeeze(model2.predict(x2train))
# define var
print('##### MLPNN for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, pred_ann2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, pred_ann2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, pred_ann2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, pred_ann2)))

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
    ax.set_xlabel('Measured SDD (m)',fontsize=18)
    ax.set_ylabel('Predicted SDD (m)',fontsize=18)
    ax.plot(np.arange(-5,30), np.arange(-5,30), 'k','--')
    bar = fig.colorbar(ax1, ax=ax)
    bar.set_label('Density', labelpad=-25, y=1.06, rotation=0, fontsize=12)#, rotation=270)
    return ax1, ax,fig

# Train
ax1, ax, fig = density_scatter(l8_train_Y, pred_ann2_train, bins = [10,10])
#ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, pred_ann2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, pred_ann2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, pred_ann2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, pred_ann2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(f1)MLPNN: Training', fontsize=24)
plt.savefig('MLPNN_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, pred_ann2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, pred_ann2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, pred_ann2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, pred_ann2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, pred_ann2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(f2)MLPNN: Testing', fontsize=24)
plt.savefig('MLPNN_test6.png', dpi=1000, bbox_inches='tight')

model2.save('./mlpnn2')
del model2

''' Linear Regression '''
# landsat8
linreg2 = LinearRegression()
linreg2.fit(x2train, l8_train_Y)
lr_y2 = linreg2.predict(x2valid)
lr_y2_train = linreg2.predict(x2train)
print('##### LR for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, lr_y2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, lr_y2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, lr_y2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, lr_y2)))

# Train
ax1, ax, fig = density_scatter(l8_train_Y, lr_y2_train, bins = [10,10])
#ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, lr_y2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, lr_y2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, lr_y2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, lr_y2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(a1)LR: Training', fontsize=24)
plt.savefig('LR_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, lr_y2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, lr_y2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, lr_y2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, lr_y2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, lr_y2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(a2)LR: Testing', fontsize=24)
plt.savefig('LR_test6.png', dpi=1000, bbox_inches='tight')

''' RandomForest '''
# landsat8
rf2 = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None)
rf2.fit(x2train, l8_train_Y)
rf_y2 = rf2.predict(x2valid)
rf_y2_train = rf2.predict(x2train)
print('##### RF for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, rf_y2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, rf_y2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, rf_y2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, rf_y2)))

# Train
ax1, ax, fig = density_scatter(l8_train_Y, rf_y2_train, bins = [10,10])
#ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, rf_y2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, rf_y2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, rf_y2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, rf_y2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(b1)RF: Training', fontsize=24)
plt.savefig('RF_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, rf_y2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, rf_y2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, rf_y2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, rf_y2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, rf_y2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(b2)RF: Testing', fontsize=24)
plt.savefig('RF_test6.png', dpi=1000, bbox_inches='tight')

''' Support Vector Machine '''
# landsat8
rbf_svr2 = SVR(kernel='rbf')
rbf_svr2.fit(x2train, l8_train_Y)
pred_svm2 = rbf_svr2.predict(x2valid)
pred_svm2_train = rbf_svr2.predict(x2train)
print('##### SVM for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, pred_svm2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, pred_svm2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, pred_svm2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, pred_svm2)))

# Train
ax1, ax, fig = density_scatter(l8_train_Y, pred_svm2_train, bins = [10,10])
#ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, pred_svm2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, pred_svm2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, pred_svm2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, pred_svm2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(c1)SVM: Training', fontsize=24)
plt.savefig('SVM_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, pred_svm2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, pred_svm2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, pred_svm2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, pred_svm2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, pred_svm2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(c2)SVM: Testing', fontsize=24)
plt.savefig('SVM_test6.png', dpi=1000, bbox_inches='tight')

''' Gaussian Process Regression '''
kernel = ConstantKernel()*RBF()+ConstantKernel()*WhiteKernel()+ConstantKernel()*RationalQuadratic()
# landsat8
gp2 = GaussianProcessRegressor(kernel=kernel,random_state=1).fit(x2train, l8_train_Y)
pred_gp2 = gp2.predict(x2valid)
pred_gp2_train = gp2.predict(x2train)
print('##### GPR for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, pred_gp2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, pred_gp2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, pred_gp2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, pred_gp2)))

# Train
ax1, ax, fig = density_scatter(l8_train_Y, pred_gp2_train, bins = [10,10])
#ax1, ax, fig = density_scatter(valid_y3, pred_rnn, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, pred_gp2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, pred_gp2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, pred_gp2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, pred_gp2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(d1)GPR: Training', fontsize=24)
plt.savefig('GPR_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, pred_gp2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, pred_gp2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, pred_gp2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, pred_gp2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, pred_gp2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(d2)GPR: Testing', fontsize=24)
plt.savefig('GPR_test6.png', dpi=1000, bbox_inches='tight')

# Plot train
''' ResNet '''
def identity_block(input_tensor,units):
	x = Dense(units)(input_tensor)
#	x = BatchNormalization()(x)
	x = Activation('relu')(x)
#
	x = Dense(units)(x)
#	x = BatchNormalization()(x)
	x = Activation('relu')(x)
#
	x = Dense(units)(x)
#	x = BatchNormalization()(x)
#
	x = add([x, input_tensor])
	x = Activation('relu')(x)
	return x
def dens_block(input_tensor,units):
	x = Dense(units)(input_tensor)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
#
	x = Dense(units)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
#
	x = Dense(units)(x)
	x = BatchNormalization()(x)
#
	shortcut = Dense(units)(input_tensor)
	shortcut = BatchNormalization()(shortcut)
#
	x = add([x, shortcut])
	x = Activation('relu')(x)
	return x
def ResNet50Regression(wid):
	Res_input = Input(shape=[None, None, 6])
	width = wid
	x = dens_block(Res_input,width)
	x = identity_block(x,width)
	x = identity_block(x,width)
#
	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)
#
	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)
#
	x = BatchNormalization()(x)
	x = Dense(1, activation='linear')(x)
	model = Model(inputs=Res_input, outputs=x)
	return model
# landsat8
nns = 512
model2 = ResNet50Regression(nns)
model2.compile(loss='mse', optimizer='adam', metrics=['mse'])
history2 = model2.fit(x2train.reshape(x2train.shape[0],1,1,6), l8_train_Y.reshape(l8_train_Y.shape[0],1,1), epochs=5000, batch_size=64, verbose=None, 
					callbacks=[EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='auto')], 
					validation_split=0.1)
pred_resn2 = np.squeeze(model2.predict(x2valid.reshape(x2valid.shape[0],1,1,6)))
pred_resn2_train = np.squeeze(model2.predict(x2train.reshape(x2train.shape[0],1,1,6)))
print('##### ResNet for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, pred_resn2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, pred_resn2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, pred_resn2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, pred_resn2)))

# model2.save('./resnet2')

# Train
ax1, ax, fig = density_scatter(l8_train_Y, pred_resn2_train, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, pred_resn2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, pred_resn2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, pred_resn2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, pred_resn2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(g1)ResNet: Training', fontsize=24)
plt.savefig('ResNet_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, pred_resn2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, pred_resn2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, pred_resn2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, pred_resn2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, pred_resn2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(g2)ResNet: Testing', fontsize=24)
plt.savefig('ResNet_test6.png', dpi=1000, bbox_inches='tight')

'''ANN'''
# Multi-Layer Perceptron Neural Network Model
def get1NN(train_x, train_y, nns, batch=None, learning_rate=0.001):
  print('##### N_NEURAL='+str(nns)+' #####')
  model = Sequential()
  model.add(Dense(nns,activation='relu', input_shape=(train_x.shape[1],), name='input'))
  model.add(Dense(nns,activation='relu'))
  model.add(Dense(1, name='output'))
  #optimizers = keras.optimizers.Adam(lr=learning_rate)
  model.compile(optimizer = 'adam', loss = 'mse')
#   print(model.summary())
  # my_callbacks = [EarlyStopping(monitor='val_loss',patience=10,verbose=2,mode='auto')]
  my_callbacks = [EarlyStopping(monitor='val_mse',patience=10,verbose=2,mode='auto')]
  if batch == None:
    history = model.fit(train_x, train_y,epochs=5000,verbose=0) #,batch_size=8, callbacks=my_callbacks, validation_split=0.1
  else:
    history = model.fit(train_x, train_y,epochs=5000,verbose=0,batch_size=batch)
  return model, history

# landsat8
model2, history = get1NN(x2train, l8_train_Y, 8, learning_rate=0.001)
pred_ann2 = np.squeeze(model2.predict(x2valid))
pred_ann2_train = np.squeeze(model2.predict(x2train))
# define mape
print('##### ANN for landsat8 #####')
print('R2_SCORE='+str(r2_score(l8_valid_Y, pred_ann2)))
print('MAE='+str(mean_absolute_error(l8_valid_Y, pred_ann2)))
print('RMSE='+str(mean_squared_error(l8_valid_Y, pred_ann2)**0.5))
print('MAPE='+str(mape(l8_valid_Y, pred_ann2)))

#model2.save('./ann2')
del model2
# Train
ax1, ax, fig = density_scatter(l8_train_Y, pred_ann2_train, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_train_Y, pred_ann2_train),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_train_Y, pred_ann2_train),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_train_Y, pred_ann2_train)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_train_Y, pred_ann2_train),2)),fontsize=20)
plt.yticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(e1)ANN: Training', fontsize=24)
plt.savefig('ANN_train6.png', dpi=1000, bbox_inches='tight')
# Test
ax1, ax, fig = density_scatter(l8_valid_Y, pred_resn2, bins = [10,10])
ax.text(0.5,11.0,'$\mathregular{R^2}$ = '+str(round(r2_score(l8_valid_Y, pred_ann2),2)),fontsize=20)
ax.text(0.5,10.0,'MAE = '+str(round(mean_absolute_error(l8_valid_Y, pred_ann2),2)),fontsize=20)
ax.text(0.5,9.0,'RMSE = '+str(round(mean_squared_error(l8_valid_Y, pred_ann2)**0.5,2)),fontsize=20)
ax.text(0.5,8.0,'MAPE = '+str(round(mape(l8_valid_Y, pred_ann2),2)),fontsize=20)
plt.xticks([0,2,4,6,8,10,12],[0,2,4,6,8,10,12],fontsize=12)
ax.set_title('(e2)ANN: Testing', fontsize=24)
print('Success!')
plt.savefig('ANN_test6.png', dpi=1000, bbox_inches='tight')
