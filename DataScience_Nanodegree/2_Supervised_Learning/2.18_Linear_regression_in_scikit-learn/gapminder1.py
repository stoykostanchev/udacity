# TODO: Add import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os as os
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
base_path = os.path.dirname(__file__)
bmi_life_data = pd.read_csv(os.path.join(base_path, './bmi_and_life_expectancy.csv'))

# Make and fit the linear regression model
#TODO: Fit the model and Assign it to bmi_life_model
X = bmi_life_data.loc[:, ['BMI']]
Y = bmi_life_data.loc[:,['Life expectancy']]
bmi_life_model = LinearRegression().fit(X, Y)

# Make a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(np.array([[21.07931]]))

# plt.figure()
# X_min = X.min()
# X_max = X.max()
print('..')
plt.plot([1,2,3,4])
# plt.scatter(X, Y, zorder = 3)
plt.show()