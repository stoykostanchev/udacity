# TODO: Add import statements
import os, pandas, numpy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# TODO: Load the data
base_path = os.path.dirname(__file__)
train_data = pandas.read_csv(os.path.join(base_path, 'data.csv'), header=None)
X = train_data.iloc[:, 0:train_data.shape[1] - 1]
y = train_data.iloc[:, train_data.shape[1] - 1]

# TODO: Create the standardization scaling object.
scaler = StandardScaler()

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X, y)
scaler.transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
model = lasso_reg.fit(X_scaled, y)


# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = model.coef_
print(reg_coef)