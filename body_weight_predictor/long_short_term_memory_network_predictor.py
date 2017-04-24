import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from body_weight_predictor.base_etl import BaseETL


def lstm(df_raw, look_back=5, end_period=30, show_plot=False):
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

    def minmaxscaling(df):
        # MinMaxScaling between 0 and 1 is bad when you have outliers.
        # https://stats.stackexchange.com/a/10298
        scaler = MinMaxScaler(feature_range=(0, 1))
        # min max scaler want features in the columns and samples in the rows -> ok
        df = scaler.fit_transform(df)
        return df, scaler

    # we need this dtype to normalize our data between 0 and 1
    # more on minmax scaling vs standardization
    # http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling
    df_raw = df_raw.astype('float32')
    X = df_raw.iloc[:, end_period - look_back:end_period]
    X, x_scaler = minmaxscaling(X)

    y = df_raw.iloc[:, -1]
    y, y_scaler = minmaxscaling(y)

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=0)

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

    """
    log tranformation does not make a difference (avoiding outliers should improve our model but that is not the case
    """

    # trainY = np.exp(trainY)
    # trainPredict = np.exp(trainPredict)
    # testY = np.exp(testY)
    # testPredict = np.exp(testPredict)

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


    if show_plot:
        create_graph(testY, testPredict, 'samples', 'weight', 'upper right', 'Test Dataset Pred. vs Obs.')
        create_graph(trainY, trainPredict, 'samples', 'weight', 'upper right', 'Training Dataset Pred. vs Obs.')

    return trainScore, testScore


def create_graph(observed, predicted, xlabel, ylabel, legend, title):
    plt.suptitle(title, fontsize=20)
    plt.plot(observed[0], label='observations')
    plt.plot(predicted[:, 0], label='prediction')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend)
    plt.show()

start_prediction = 30
to_predict_age_day = 35
from collections import defaultdict
best_score = defaultdict(list)
for i in range(start_prediction):
    if i > 0:
        look_back = i
        print('look_back period {}: we use period {} to {}'.format(i, start_prediction - i, start_prediction))
        etl = BaseETL('bw_ross_308.csv', start_prediction, to_predict_age_day, False)
        X_train, X_test, y_train, y_test = etl.process()
        df_raw = etl.df_raw
        train_score, test_score = lstm(df_raw, look_back, start_prediction)
        best_score['test_score'].append(test_score)
        best_score['train_score'].append(train_score)
        best_score['look_back'].append(look_back)

# print('beste test score {} RMSE with train score {}, lookback period {}'.format(best_score['test_score'],
#                                                                                 best_score['train_score'],
#
#                        best_score['look_back']))
best_score
df_score = pd.DataFrame(best_score).set_index('look_back')
plt.suptitle('scores', fontsize=20)
plt.plot(df_score['test_score'], label='test')
plt.plot(df_score['train_score'], label='train')
plt.xlabel('lookback')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.show()
print(best_score)