import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

def lstm(df_raw, look_back=5, end_period=30, show_plot=True):
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
    model.add(LSTM(4, input_shape=(None, look_back))) # todo - investigate why 4 nodes. check initial url of lstm
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


def plot_best_score(best_score, start_prediction, figsize=(15,15)):
    df_score = pd.DataFrame(best_score).set_index('look_back')
    plt.suptitle('scores', fontsize=20)
    plt.figure(figsize=figsize)
    plt.plot(df_score['test_score'], label='test')
    plt.plot(df_score['train_score'], label='train')
    plt.xlabel('lookback')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.show()
    index_min_rmse_test = best_score['test_score'].index(min(best_score['test_score']))
    print(
        'best test score {} RMSE with train score {}, the lookback period we use starting from day {} is {} days'.format(
            best_score['test_score'][index_min_rmse_test],
            best_score['train_score'][index_min_rmse_test], start_prediction,
            best_score['look_back'][index_min_rmse_test]))


def find_best_look_back_period(df=None, start_prediction = 30, to_predict_age_day = 35):
    """
    Predict the body weight on day "to_predict_age_day" starting from day "start_prediction".
    
    :param df: 
    :param start_prediction: 
    :param to_predict_age_day: 
    :return: 
    """
    if df is None:
        # we import here because if we run from notebook we cannot do this import
        from base_etl import BaseETL
        etl = BaseETL('bw_ross_308.csv', start_of_predicition=start_prediction,
                  dependend_variable_period=to_predict_age_day, show_plot=False)
        etl.process()
        df = etl.df_raw
    else:
        df = df.T # if do not transpose we also get a very good score.
                  # which is very weird ....
    scores = defaultdict(list)
    for i in range(start_prediction):
        if i > 0:
            look_back = i
            train_score, test_score = lstm(df, look_back, start_prediction, show_plot=False)
            scores['test_score'].append(test_score)
            scores['train_score'].append(train_score)
            scores['look_back'].append(look_back)
    return scores


def get_df_for_prediction(start_prediction=30, to_predict_age_day=35):
    from base_etl import BaseETL
    etl = BaseETL('bw_ross_308.csv', start_of_predicition=start_prediction,
                  dependend_variable_period=to_predict_age_day, show_plot=False)
    etl.process()
    df = etl.df_raw
    return df


def predict_value_using_a_lookback_period(lookback, start_prediction, to_predict_age_day = 35):
    score = defaultdict(list)
    for i in range(start_prediction):
        if i > 1:
            df = get_df_for_prediction(start_prediction=i)
            if i < lookback:
                train_score, test_score = lstm(df, i - 1, i, show_plot=False) # if we have 2 days as input we can only use these 2 days as lookback.
            else:
                train_score, test_score = lstm(df, lookback, i, show_plot=False)
            score['test_score'].append(test_score)
            score['train_score'].append(train_score)
            score['look_back'].append(lookback)
    return score

def predict_value_while_finding_best_lookback_period(start_prediction, to_predict_age_day = 35):
    best_score = {}
    for i in range(start_prediction):
        if i > 1:
            scores = find_best_look_back_period(start_prediction=i, to_predict_age_day=35)
            index_min_rmse_test = scores['test_score'].index(min(scores['test_score']))
            agg_scores = {}
            agg_scores['test_score'] = scores['test_score'][index_min_rmse_test]
            agg_scores['train_score'] = scores['train_score'][index_min_rmse_test]
            agg_scores['look_back'] = scores['look_back'][index_min_rmse_test]
            best_score[i] = agg_scores
    return best_score
