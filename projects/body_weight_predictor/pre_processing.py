import pandas as pd
import matplotlib.pyplot as plt


class PreProcessing(object):
    def __init__(self, file_path, start_of_predicition, dependend_variable_period, show_plot=False):
        self.file_path = file_path  # 'bw_ross_308.csv'
        self.start_of_prediction = start_of_predicition
        self.dependend_variable_period = dependend_variable_period
        self.df_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.show_plot = show_plot

    def process(self):
        self.extract()
        self.data_cleaning()
        self.load()
        if self.show_plot:
            self.plot()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_df(self, df, index='age_days', show_legend=False):
        if self.show_plot:
            df = df.set_index(index, drop=True)
            df.plot(legend=True)
            plt.show()

    def extract(self):
        # READ IN DATA
        # -----------------------
        self.df_raw = pd.read_csv(self.file_path).sort_values('age_days').reset_index(drop=True)
        return self.df_raw

    def basic_cleaning(self):
        print('initial shape df : {}'.format(self.df_raw.shape))

        # remove empty columns
        self.df_raw = self.df_raw.dropna(axis=1, how='all')

        print('removed rounds with empty body weight. shape of df {}'.format(self.df_raw.shape))

        # we can have only 1 value per day for bw.
        self.df_raw = self.df_raw.drop_duplicates(subset='age_days')

        print('removed duplicates days. we can only have 1 value per day. '
              'we keep the first value we encounter. shape of df {}'.format(self.df_raw.shape))

        # remove data after 35 days
        self.df_raw = self.df_raw[self.df_raw['age_days'] <= self.dependend_variable_period]
        self.df_raw = self.df_raw.set_index('age_days', drop=True)

        print('removed values for days after our dependend variable {} '
              '- shape of df {}.'.format(self.dependend_variable_period, self.df_raw.shape))

        return self.df_raw

    def data_cleaning(self):
        # DATA PREPROCESSING
        # -----------------------

        self.df_raw = self.basic_cleaning()

        self.plot_df(self.df_raw)

        # remove rounds with no initial data
        mask_no_initial_data = self.df_raw.fillna(method='ffill').notnull()
        self.df_raw = self.df_raw.loc[:, mask_no_initial_data.all(axis=0)]

        print('we cannot use rounds with no initial data. shape of df {}'.format(self.df_raw.shape))

        # remove rounds which have NaN before 35 days
        mask_no_end_data = self.df_raw.fillna(method='backfill').notnull()
        self.df_raw = self.df_raw.loc[:, mask_no_end_data.all(axis=0)]

        print('we remove rounds which have no data before our dependend variable. shape of df is {}'.format(self.dependend_variable_period, self.df_raw.shape))

        # remove all columns that have same value
        self.df_raw = self.df_raw.loc[:, self.df_raw.all(axis=0)]

        print('we remove all columns with the same value'.format(self.df_raw.shape))

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

        self.df_raw = duplicate_values_in_series(self.df_raw)

        print('we remove series with 25% same values in the series. df shape is {}'.format(self.df_raw.shape))

        # drop columns which have extreme values in the beginning of the round.
        # sometimes the regulator is not reset before the next flock starts.
        # we ignore these extreme values
        # we keep the columns which have no extreme values in the first 5 days.

        self.df_raw = self.df_raw.loc[:, ~((self.df_raw.iloc[:5, :] > 500).any())]

        print('drop columns which have a high value ( bw of 500 g ) within the first 5 days. df shape is {}'.format(self.df_raw.shape))

    def load(self):
        self.df_raw = self.df_raw.T


        # get X, Y

        X = self.df_raw.iloc[:, 0: self.start_of_prediction]
        y = self.df_raw.iloc[:, -1]

        # Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot(self):
        # plot the data with age in x axis and weight on the y axis.
        self.df_raw.T.plot()
        plt.show()
