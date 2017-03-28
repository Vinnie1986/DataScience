import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# READ IN DATA
# -----------------------
df_raw = pd.read_csv('bw_ross_308.csv').sort('age_days').reset_index(drop=True)
df_raw.head()

# DATA PREPROCESSING
# -----------------------
print df_raw.shape
# remove empty columns
df_raw = df_raw.dropna(axis=1, how='all')

print df_raw.shape

# we can have only 1 value per day for bw.
df_raw = df_raw.drop_duplicates(subset='age_days')

print df_raw.shape

# remove data after 35 days
df_raw = df_raw[df_raw['age_days'] <= 35]
df_raw = df_raw.set_index('age_days', drop=True)

print df_raw.shape

# remove rounds with no initial data
mask_no_initial_data = df_raw.fillna(method='ffill').notnull()
df_raw = df_raw.loc[:, mask_no_initial_data.all(axis=0)]

print df_raw.shape

# remove rounds which have NaN before 35 days
mask_no_end_data = df_raw.fillna(method='backfill').notnull()
df_raw = df_raw.loc[:, mask_no_end_data.all(axis=0)]

print df_raw.shape

# remove all columns that have same value
df_raw = df_raw.loc[:, df_raw.all(axis=0)]

print df_raw.shape


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


raw_df = duplicate_values_in_series(df_raw)

print df_raw.shape

# drop columns which have extreme values in the beginning of the round to nan.
# sometimes the regulator is not reset before the next flock starts.
# we ignore these extreme values

print df_raw.shape
# we keep the columns which have no extreme values in the first 5 days.
df_raw = df_raw.loc[:, ~((df_raw.iloc[:5, :] > 500).any())]
print df_raw.shape

df_raw.plot()
plt.show()


df_raw = df_raw.T

print df_raw.describe()

# get X, Y

X = df_raw.iloc[:, 0:5]
y = df_raw.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#################################
#
# RANDOM FOREST REGRESSION
#
#################################


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)
test_predicted = regressor.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((test_predicted - y_test) ** 2))

# error std. dev.
print("error std. dev.: %.2f"
      % np.sqrt(np.mean((test_predicted - y_test) ** 2)))

# Plot outputs

fig, ax = plt.subplots()
ax.scatter(y_test, test_predicted)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



