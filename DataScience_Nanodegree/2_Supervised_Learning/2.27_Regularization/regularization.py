# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
import os


# Assign the data to predictor and outcome variables
# TODO: Load the data
base_path = os.path.dirname(__file__)
train_data = pd.read_csv(os.path.join(base_path, './data.csv'), header=None)

X = train_data.iloc[:,0:train_data.shape[1] - 1]
y = train_data.iloc[:,train_data.shape[1] - 1]

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
model = lasso_reg.fit(X, y)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = model.coef_
print(reg_coef)