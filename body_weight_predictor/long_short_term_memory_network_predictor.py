import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from body_weight_predictor.etl import ETL

X_train, X_test, y_train, y_test = ETL('bw_ross_308.csv', 5, 35).process()



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


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
# dataframe = pd.read_csv('international-airline-passengers.csv', sep=';', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# normalize the dataset

# handle inf and nan values. if we normalize the values they cannot handle nan and inf
df_raw = df_raw.replace([np.inf, -np.inf], np.nan)
for column in df_raw:
    df_raw[column] = df_raw[column].fillna(df_raw[column].mean())
df_raw = df_raw.astype('float32')
# scaler = MinMaxScaler(feature_range=(0, 1))
# df_raw = scaler.fit_transform(df_raw)
# split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
X = df_raw.iloc[:,0:5]
x_scaler = MinMaxScaler(feature_range=(0, 1))
X = x_scaler.fit_transform(X)

y = df_raw.iloc[:,-1]
y_scaler = MinMaxScaler(feature_range=(0, 1))
y = y_scaler.fit_transform(y)

dataset = df_raw
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)


look_back = 5
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=10, batch_size=1, verbose=2)
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
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
#fix this scaler
# plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
