import matplotlib.pyplot as plt
import numpy as np

from projects.body_weight_predictor import BaseETL

X_train, X_test, y_train, y_test = BaseETL('bw_ross_308.csv', 5, 35).process()

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



