import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import sort
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import json
import mca
import pandas

def main():
    adj = sp.load_npz('adj.npz')
    feat = np.load('features.npy')
    L = np.load('labels.npy')
    splits = json.load(open('splits.json'))
    idx_train, idx_test = splits['idx_train'], splits['idx_test']
    idx_train_s, idx_test_s = set(idx_train), set(idx_test)

    Xtrain, Xtest = np.zeros((len(idx_train), feat.shape[1])), np.zeros((len(idx_test), feat.shape[1]))

    # Populate the training and testing datasets
    for i, v in enumerate(idx_train):
        Xtrain[i, :] = feat[v]
    for i, v in enumerate(idx_test):
        Xtest[i, :] = feat[v]

    #df = pandas.DataFrame(Xtrain, columns=[str(i) for i in range(len(Xtrain[0]))])
    #x = mca.MCA(df)

    p = logistic_pca(feat, num_iter=1)
    
    b=1


def logistic_pca(X, num_components=None, num_iter=32):
    """Logistic principal component analysis (PCA).
    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    num_iter : int, default=32
        Number iterations for fitting model.
    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Estimated projection matrix.
    mu : (num_components, num_samples) ndarray
        Estimated latent variables.
    b : (num_dimensions, 1) ndarray
        Estimated bias.
    Reference
    ----------
    Tipping, Michael E. "Probabilistic visualisation of high-dimensional binary data." 
    Advances in neural information processing systems (1999): 592-598.
    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    # shorthands
    N = num_samples
    D = num_dimensions
    K = num_components
    # initialise
    I = np.eye(K)
    W = np.random.randn(D, K)
    mu = np.random.randn(K, N)
    b = np.random.randn(D, 1)    
    C = np.repeat(I[:, :, np.newaxis], N, axis=2)
    xi = np.ones((N, D))  # the variational parameters
    # functions
    sig = lambda x: 1/(1 + np.exp(-x))
    lam = lambda x: (0.5 - sig(x))/(2*x)
    # fit model
    for iter in range(num_iter):
        # 1.obtain the sufficient statistics for the approximated posterior 
        # distribution of latent variables given each observation
        for n in range(N):
            # get sample
            x_n = X[n, :][:, None]
            # compute approximation
            lam_n = lam(xi[n, :])[:, None]
            # update
            C[:, :, n] = inv(I - 2*mm(W.T, lam_n*W))
            mu[:, n] = mm(C[:, :, n], mm(W.T, x_n - 0.5 + 2*lam_n*b))[:, 0]
        # 2.optimise the variational parameters in in order to make the 
        # approximation as close as possible
        for n in range(N):
            # posterior statistics
            z = mu[:, n][:, None]
            E_zz = C[:, :, n] + mm(z, z.T)
            # xi squared
            xixi = np.sum(W*mm(W, E_zz), axis=1, keepdims=True) \
                   + 2*b*mm(W, z) + b**2
            # update
            xi[n, :] = np.sqrt(np.abs(xixi[:, 0]))
        # 3.update model parameters
        E_zhzh = np.zeros((K + 1, K + 1, N))
        for n in range(N):
            z = mu[:, n][:, None]
            E_zhzh[:-1, :-1, n] = C[:, :, n] + mm(z, z.T)
            E_zhzh[:-1, -1, n] = z[:, 0]
            E_zhzh[-1, :-1, n] = z[:, 0]
            E_zhzh[-1, -1, n] = 1
        E_zh = np.append(mu, np.ones((1, N)), axis=0)
        for i in range(D):
            # compute approximation
            lam_i = lam(xi[:, i])[None][None]
            # gradient and Hessian
            H = np.sum(2*lam_i*E_zhzh, axis=2)
            g = mm(E_zh, X[:, i] - 0.5)
            # invert
            wh_i = -solve(H, g[:, None])
            wh_i = wh_i[:, 0]
            # update
            W[i, :] = wh_i[:K]
            b[i] = wh_i[K]

    return W, mu, b

def _get_num_components(num_components, num_samples, num_dimensions):
    """Get number of components (clusters).
    """
    if num_components is None:
        num_components = min(num_samples, num_dimensions)    

    return num_components 

if __name__ == "__main__":
    main()
