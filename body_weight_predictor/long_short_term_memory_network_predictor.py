import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from body_weight_predictor.etl import ETL

etl = ETL('bw_ross_308.csv', 30, 35)
X_train, X_test, y_train, y_test = etl.process()

df_raw = etl.df_raw

########################################
#
# LSTM network
#
########################################

# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)

# handle inf and nan values. if we normalize the values they cannot handle nan and inf
# replace inf by nan
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
for column in df_raw:
    # replace nan with mean of the column
    df_raw[column] = df_raw[column].fillna(df_raw[column].mean())

# we need this dtype to normalize our data between 0 and 1
# more on minmax scaling vs standardization
# http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling
df_raw = df_raw.astype('float32')
X = df_raw.iloc[:, 0:5]
x_scaler = MinMaxScaler(feature_range=(0, 1))
X = x_scaler.fit_transform(X)

y = df_raw.iloc[:, -1]
y_scaler = MinMaxScaler(feature_range=(0, 1))
y = y_scaler.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)

look_back = 5

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
# http://deeplearning.net/tutorial/lstm.html
model = Sequential()
model.add(LSTM(4, input_shape=(None, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = y_scaler.inverse_transform(trainPredict)
trainY = y_scaler.inverse_transform([trainY])
testPredict = y_scaler.inverse_transform(testPredict)
testY = y_scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# create graph for test data set
plt.suptitle('Test Dataset Pred. vs Obs.', fontsize=20)
plt.plot(testPredict[:, 0], label='prediction')
plt.plot(testY[0], label='observations')
plt.ylabel('weight')
plt.xlabel('samples')
plt.legend(loc='upper right')
plt.show()
