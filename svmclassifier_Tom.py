import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()


#choose a seed
seed = 2027
np.random.seed(seed)
# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)

N,d = X.shape
Ntest = 100
Xtrain = X[0:-Ntest,:]
ytrain = y[0:-Ntest]
Xtest = X[-Ntest:,:]
ytest = y[-Ntest:]
#print("y", ytrain[0:10])
print("x", Xtrain)

def svmsubgradient(Theta, x, y):
    """
    Returns a subgradient of the objective empirical hinge loss.
    
    The inputs are Theta, of size d-by-K, where K is the number of classes,
    x of size d, and y an integer in {0, 1, ..., 9}.
    """
    G = np.zeros(Theta.shape)
    X_scores = x @ Theta
    J_star = np.argmax(X_scores)
    if J_star == y:
        X_scores[J_star] = -np.inf
        J_star = np.argmax(X_scores)
    else:
        pass

    g_j = 1 + x @ Theta[:,J_star] - x @ Theta[:,y]
    if g_j > 0:
        G[:, J_star] = x
        G[:, y] = -x
    else:
        pass #since G is already zeros

    return(G)

def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000):
    """
    Performs maxiter iterations of projected stochastic gradient descent
    on the data contained in the matrix Xtrain, of size n-by-d, where n
    is the sample size and d is the dimension, and the label vector
    ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
    classification matrices Theta and mean_Theta, where the first is the final
    point of SGD and the second is the mean of all the iterates of SGD.
    
    Each iteration consists of choosing a random index from n and the
    associated data point in X, taking a subgradient step for the
    multiclass SVM objective, and projecting onto the Euclidean ball
    The stepsize is init_stepsize / sqrt(iteration).
    """
    K = 10
    NN, dd = Xtrain.shape
    print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K
    ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT


    return Theta, mean_Theta

def classify(Xdata, Theta):
    """
    Takes in an N-by-d data matrix Adata, where d is the dimension and N
    is the sample size, and a classifier X, which is of size d-by-K,
    where K is the number of classes.
    
    Returns a vector of length N consisting of the predicted digits in
    the classes.
    """
    scores = np.matmul(Xdata, Theta)
    inds = np.argmax(scores, axis = 1)
    return(inds)


def main():
    l2_radius = 40.0
    M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
    init_stepsize = l2_radius/M_raw
    maxiter = 40000
    Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius)
    print('Error rate')
    print(np.sum(np.not_equal(classify(Xtest, mean_Theta),ytest)/Ntest))

def test():
    svmsubgradient(1, Xtrain[0:10],ytrain[0:10])

test()