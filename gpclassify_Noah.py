"""
=====================================================
Gaussian process classification (GPC) 
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor

#choose a seed
seed = 2027
np.random.seed(seed)

# import some data to play with
digits = datasets.load_digits()

X = digits.data / 16.0
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape 

Ntrain_list = [20, 50, 100, 500, 1000, 1500]
Ntest = 100

Xtest = X[-Ntest:,:]
ytest = y[-Ntest:]

#kernel = 1.0 * RBF([1.0]) # isotropic kernel
# kernel = DotProduct(1.0) 
#GaussianProcessClassifier performs hyperparameter estimation, this means that the the value specified above may not be the final hyperparameters
#If you dont want it to do hyperparameter optimization set optimizer = None like this GaussianProcessClassifier(kernel=kernel,optimizer = None).fit(Xtrain, ytrain)
#You can check the final hyperparameters with gpc_rbf.kernel_.get_params()['kernels']

kernels = {
    "RBF (l=1)": 1.0 * RBF(length_scale=1.0),
    "RBF (l=2)": 1.0 * RBF(length_scale=2.0),
    "DotProduct (s0=1)": DotProduct(sigma_0=1.0),
    "DotProduct (s0=2)": DotProduct(sigma_0=2.0),
}

for kernel_name, kernel in kernels.items():
    test_errors = []
    train_errors = []
    for Ntrain in Ntrain_list:
        Xtrain = X[0:Ntrain, :]
        ytrain = y[0:Ntrain]

        
        gpc = GaussianProcessClassifier(kernel=kernel, optimizer=None).fit(Xtrain, ytrain)
        yp_train = gpc.predict(Xtrain)
        train_error_rate = np.mean(np.not_equal(yp_train,ytrain))
        yp_test = gpc.predict(Xtest)
        test_error_rate = np.mean(np.not_equal(yp_test,ytest))

        train_errors.append(train_error_rate)
        test_errors.append(test_error_rate)

        print(kernel_name, Ntrain, train_error_rate, test_error_rate)
    plt.plot(Ntrain_list, test_errors, marker="o", label=kernel_name)

plt.xlabel("Number of training samples")
plt.ylabel("Test error rate")
plt.title("Gaussian Process Classification")
plt.legend()
plt.grid()
plt.show()



for kernel_name, kernel in kernels.items():

    test_errors = []
    train_errors = []

    for Ntrain in Ntrain_list:

        # Select training data
        Xtrain = X[0:Ntrain, :]
        ytrain = y[0:Ntrain]

        # Convert labels to one-hot vectors
        Ytrain = np.eye(10)[ytrain]

        # Train Gaussian Process Regressor
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None).fit(Xtrain, Ytrain)

        # Predict training data
        Yp_train = gpr.predict(Xtrain)
        yp_train = np.argmax(Yp_train, axis=1)

        train_error_rate = np.mean(
            np.not_equal(yp_train, ytrain)
        )

        # Predict test data
        Yp_test = gpr.predict(Xtest)
        yp_test = np.argmax(Yp_test, axis=1)

        test_error_rate = np.mean(
            np.not_equal(yp_test, ytest)
        )

        # Store results
        train_errors.append(train_error_rate)
        test_errors.append(test_error_rate)

        print(kernel_name, Ntrain, train_error_rate, test_error_rate)

    plt.plot(Ntrain_list, test_errors, marker="o", label=kernel_name)

plt.xlabel("Number of training samples")
plt.ylabel("Test error rate")
plt.title("Gaussian Process Regression Classification")
plt.legend()
plt.grid()
plt.show()

