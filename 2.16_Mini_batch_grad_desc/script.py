#https://en.wikipedia.org/wiki/Partial_derivative
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
# Setting a random seed, feel free to change it and see different solutions.
np.random.seed(42)

# TODO: Fill in code in the function below to implement a gradient descent
# step for linear regression, following a squared error rule. See the docstring
# for parameters and returned variables.
def MSEStep(X, y, W, b, learn_rate = 0.005):
    """
    This function implements the gradient descent step for squared error as a
    performance metric.
    
    Parameters
    X : array of predictor features
    y : array of outcome values
    W : predictor feature coefficients
    b : regression function intercept
    learn_rate : learning rate

    Returns
    W_new : predictor feature coefficients following gradient descent step
    b_new : intercept following gradient descent step
    --------

    H(x) = W0 * x + b [aka b] // This is our hypothesis - e.g. we THINK a line will do, e.g. a linear eq. will do
    // Note - some notebooks depict Wi and b as 

    y`1 = H(X1)

    Y` = y`1, y`2, ... , y`m

    Mean Sqrd Err = 1/2m Z{i=1, i=m} (y - y`)^2 -> 'Mean Sqrd Err' = "J" as well

    Mean Sqrd Err = r(X, Y) - H is implied? = 1/2 * X.length * [ [1...m].sum( [ H(Xi) - Yi ]^2 ) ]  

    GD = general algorythm to minimize a fn. In our case - the MSE

    GD: W`i = Wi - learning_rate * ..d/dWi..? of MSE    

    
    GD step: wi  -> wi - (learn. rate * d/dwi of Err) 

    d/dwi of Err :
    W0 = W0 - [  1/2m Z{i=1, i=m} (y - y`)^2 ]

    https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd
    http://mccormickml.com/2014/03/04/gradient-descent-derivation/
    """

    # Fill in code

    #-----
    # MSE = E(w1, w2) =  1/2m * Z{1:m} (y`i - y) ^2
    # ===> y'i = w1 * xi + w2
    # In order to minimise E, we need to take the derivataives with respect to w1 and w2 and set them to 0
    # ===> d/dw1 E(w1, w2) = 1/m * Z{1:m}(the values of the derivatives in respect to W1) =
    #     =  - (y - y`)x = y`x - yx
    # ===> d/dw2 E(w1, w2) = 1/m * Z{1:m}(the values of the derivatives in respect to W2)
    #     =  - (y - y`) = y` - y
    # Update rule:
    # ===> wi = wi - a * d/dWi Err
    #-------------


    # CALCULATE THE DERIVATIVE FOR THE MSE in respect to w1 and in respect to b
    m = X.shape[0] # -> 20
    w1 = W[0]
    w2 = b
    derivativeMSEw1 = 0
    derivativeMSEb = 0

    for i in range(0, X.shape[0]):
        xi = X[i][0] # -> NOTE: X IS AN ARRAY OF ARRAYS for some reason
        yi = y[i]
        yP = w1 * xi + w2
        derivativeW1 = (yi - yP * xi) * xi
        derivativeB = - (yi - yP)
        # The actual derivative is 1/m * sum(the values of the derivatives in respect to W1 / b)
        derivativeMSEw1 += derivativeW1 / m
        derivativeMSEb += derivativeB / m

    b_new = b - learn_rate * derivativeMSEw1
    W_new = W[0] - learn_rate * derivativeMSEb

    #After the first batch, the expected coefficients were [0.123, 0.205]. Your coefficients were [0.010, -0.006]
    #After the last batch, the expected coefficients were [0.448, 1.829]. Your coefficients were [0.245, -0.067]

    return np.array([W_new]), b_new



# The parts of the script below will be run when you press the "Test Run"
# button. The gradient descent step will be performed multiple times on
# the provided dataset, and the returned list of regression coefficients
# will be plotted.
def miniBatchGD(X, y, batch_size = 20, learn_rate = 0.005, num_iter = 25):
    """
    This function performs mini-batch gradient descent on a given dataset.

    Parameters
    X : array of predictor features
    y : array of outcome values
    batch_size : how many data points will be sampled for each iteration
    learn_rate : learning rate
    num_iter : number of batches used

    Returns
    regression_coef : array of slopes and intercepts generated by gradient
      descent procedure
    """
    n_points = X.shape[0]
    W = np.zeros(X.shape[1]) # coefficients
    b = 0 # intercept
    
    # run iterations
    regression_coef = [np.hstack((W,b))]
    for _ in range(num_iter):
        batch = np.random.choice(range(n_points), batch_size)
        X_batch = X[batch,:]
        y_batch = y[batch]
        W, b = MSEStep(X_batch, y_batch, W, b, learn_rate)
        regression_coef.append(np.hstack((W,b)))
    
    return regression_coef


if __name__ == "__main__":
    # perform gradient descent
    data = np.loadtxt('data.csv', delimiter = ',')
    X = data[:,:-1]
    y = data[:,-1]
    regression_coef = miniBatchGD(X, y)
    
    # plot the results
    import matplotlib.pyplot as plt
    
    plt.figure()
    X_min = X.min()
    X_max = X.max()
    counter = len(regression_coef)
    for W, b in regression_coef:
        counter -= 1
        color = [1 - 0.92 ** counter for _ in range(3)]
        plt.plot([X_min, X_max],[X_min * W + b, X_max * W + b], color = color)
    plt.scatter(X, y, zorder = 3)
    plt.show()