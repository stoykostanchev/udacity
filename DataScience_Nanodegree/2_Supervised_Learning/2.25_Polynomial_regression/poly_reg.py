# TODO: Add import statements
import pandas as pd
import os as os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Assign the data to predictor and outcome variables
# TODO: Load the data
base_path = os.path.dirname(__file__) 
train_data = pd.read_csv(os.path.join(base_path, './data.csv'))

X = train_data['Var_X'].values.reshape(-1, 1)
y = train_data['Var_Y']

print(X)
# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(2)
X_poly = poly_feat.fit_transform(X)

# print(X_poly.shape)
# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression().fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!