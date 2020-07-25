import numpy as np
import pandas as pd

def get_dis_matrix(data):
    """
    Calculate the distance matrix of the samples
    data:raw data with n rows and m columns, each row represents a sample and each column represents a feature
    dismat:distance matrix of all the samples
    """
    data = np.array(data)
    nNum = len(data)
    dismat = np.zeros((nNum, nNum))
    for i in range(nNum):
        dismat[i,:] = np.sqrt(np.sum(np.power(data[i]-data, 2), axis=1)).T
    return dismat

def getW(data, K, sig = 1.0, method='knearest'):
    """
    Get similarity matrix
    K:the parameter of KNN
    sig:sigma
    method:{'knearest','gauss'},default:'knearest'
        the strategy to get the similarity matrix
    W:similarity matrix
    return:W
    """
    dismat = get_dis_matrix(data)
    nNum = len(dismat)
    if method == 'knearest':
        W = np.zeros((nNum, nNum))
        for idx, val in enumerate(dismat):
            W[idx][np.argsort(val)[1:K+1]] = np.exp(-dismat[idx,[np.argsort(val)[1:K+1]]]/2/(sig**2))
        temp_W = W.T
        W = (W + temp_W) / 2
    elif method == 'gauss':
        W = np.exp(-sig*dismat**2)
    return W

def getD(W):
    """
    Get degree matrix
    """
    return np.diag(np.sum(W, axis=1))

def getL(D, W):
    """
    Get normed laplace matrix:D^(-1/2)*L*D^(-1/2)
    L:laplace matrix
    """
    L = D - W
    negsqrtD = np.diag(np.power(np.diag(D), -1/2))
    L = np.dot(np.dot(negsqrtD,L),negsqrtD)
    return L

def getEigen(L, cluster_num):
    """
    Get eigenvector as feature, return the first cluster_num eigenvector as the main vectors
    """
    eigval, eigvec = np.linalg.eig(L)
    ix = np.argsort(eigval)[0:cluster_num]
    return eigvec[:, ix]