import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play wit

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
def train_data(N_train):
    
    Xtrain = X[0:N_train, :]
    ytrain = y[0:N_train]

    Xtest = X[-Ntest:, :]
    ytest = y[-Ntest:]
    print(Xtrain.shape)
    print(Xtest.shape)

    #print(Xtest)
    #print(ytest)
    #print("y", ytrain[0:10])
    #print("x", Xtrain)
    return Xtrain, ytrain, Xtest, ytest

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

def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000, alpha = 0.5):

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
    #print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K

    for iter in range(1,maxiter + 1):
        i = np.random.randint(0, NN)

        x = Xtrain[i]
        y = ytrain[i]
        G = svmsubgradient(Theta, x, y)

        lr = init_stepsize / (iter ** alpha)
        
        Theta = Theta - lr*G
        Theta_norm = np.linalg.norm(Theta)

        if Theta_norm > l2_radius:
            Theta = l2_radius/Theta_norm * Theta
        else:
            pass

        mean_Theta +=Theta


    mean_Theta = mean_Theta/maxiter

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


def main(Xtrain, ytrain, Xtest, ytest, N_train, alpha):
    

    l2_radius = 40.0
    M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
    init_stepsize = l2_radius/M_raw
    maxiter = 40000
    
    
    Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius, alpha)
    
    # Find an incorrectly classified image
    predictions = classify(Xtest, mean_Theta)
    wrong_indices = np.where(predictions != ytest)[0]

    if len(wrong_indices) > 0:
        i = wrong_indices[0]

        plt.figure(figsize=(4, 4))
        plt.imshow(Xtest[i].reshape(8, 8), cmap='gray')
        plt.title(f"True digit: {ytest[i]} | Predicted: {predictions[i]}")
        plt.axis('off')
        plt.show()
    else:
        print("All test images were classified correctly!")

    print('Error rate for N_train = ', N_train)
    not_equal_mean = np.sum(np.not_equal( classify(Xtest, mean_Theta), ytest) / Ntest)
    not_equal = np.sum(np.not_equal( classify(Xtest, Theta), ytest) / Ntest)
    print("Wrong classification percentage for theta mean ", not_equal_mean)
    print("Wrong classification percentage for theta ", not_equal)


    train_error_mean = np.sum(np.not_equal( classify(Xtrain, mean_Theta), ytrain) / N_train)
    train_error = np.sum(np.not_equal( classify(Xtrain, Theta), ytrain) / N_train)

    print("Training error: ", train_error_mean)
    print("Training error mean: ", train_error)
    #print("Test error:", test_error)

    #print("Predicted:", predictions)
    #print("Actual:   ", ytest)

    #print("Incorrect predictions:", np.sum(predictions != ytest))
    #print("Accuracy:", np.mean(predictions == ytest) * 100, "%")
    return not_equal_mean, not_equal


Xtrain, ytrain, Xtest, ytest = train_data(1500)

main(Xtrain, ytrain, Xtest, ytest,1500, 0.5)
