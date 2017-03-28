import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# READ IN DATA
# -----------------------
df_raw = pd.read_csv('bw_ross_308.csv').sort('age_days').reset_index(drop=True)
df_raw.head()

# DATA PREPROCESSING
# -----------------------
print(df_raw.shape)
# remove empty columns
df_raw = df_raw.dropna(axis=1, how='all')

print(df_raw.shape)

# we can have only 1 value per day for bw.
df_raw = df_raw.drop_duplicates(subset='age_days')

print(df_raw.shape)

# remove data after 35 days
df_raw = df_raw[df_raw['age_days'] <= 35]
df_raw = df_raw.set_index('age_days', drop=True)

print(df_raw.shape)

# remove rounds with no initial data
mask_no_initial_data = df_raw.fillna(method='ffill').notnull()
df_raw = df_raw.loc[:, mask_no_initial_data.all(axis=0)]

print(df_raw.shape)

# remove rounds which have NaN before 35 days
mask_no_end_data = df_raw.fillna(method='backfill').notnull()
df_raw = df_raw.loc[:, mask_no_end_data.all(axis=0)]

print(df_raw.shape)

# remove all columns that have same value
df_raw = df_raw.loc[:, df_raw.all(axis=0)]

print(df_raw.shape)


def duplicate_values_in_series(df, percentage_same_values=0.25):
    '''
    remove columns that have the same value for more than the percentage_same_values.
    if a column satisfies this condition we remove it.

    if we have a record with e.g. number of values equal to 4 that is more than percentage_same_value,
    we do not want this in out training/testing set.

    :param df:
    :param percentage_same_values:
    :return:
    '''
    for column in df:
        duplicates_mask = df[column].value_counts()
        _len = float(len(duplicates_mask))
        duplicates = duplicates_mask.max()
        if duplicates / _len >= percentage_same_values:
            del df[column]
    return df


df_raw = duplicate_values_in_series(df_raw)

print(df_raw.shape)

# drop columns which have extreme values in the beginning of the round to nan.
# sometimes the regulator is not reset before the next flock starts.
# we ignore these extreme values

print(df_raw.shape)
# we keep the columns which have no extreme values in the first 5 days.
df_raw = df_raw.loc[:, ~((df_raw.iloc[:5, :] > 500).any())]
print(df_raw.shape)

df_raw.plot()
plt.show()


df_raw = df_raw.T

print(df_raw.describe())

# get X, Y

X = df_raw.iloc[:, 0:5]
y = df_raw.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



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
