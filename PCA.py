import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import json


def main():
    adj = sp.load_npz('adj.npz')
    feat = np.load('features.npy')
    L = np.load('labels.npy')
    splits = json.load(open('splits.json'))
    idx_train, idx_test = splits['idx_train'], splits['idx_test']
    idx_train_s, idx_test_s = set(idx_train), set(idx_test)

    pca2D = PCA(n_components=2)
    pca3D = PCA(n_components=3)
    pca = PCA(n_components=50)

    Xtrain, Xtest = np.zeros((len(idx_train), feat.shape[1])), np.zeros((len(idx_test), feat.shape[1]))

    # Populate the training and testing datasets
    for i, v in enumerate(idx_train):
        Xtrain[i, :] = feat[v]
    for i, v in enumerate(idx_test):
        Xtest[i, :] = feat[v]

    #groups = []

    #for i in range(7):
    #    local = []
    #    for j in range(len(L)):
    #        if i == L[j]:
    #            local.append(Xtrain[j])
    #    groups.append(np.array(local))
    #groups = np.array(groups)
    #pcas = []
    #for group in groups:
    #    pca = PCA(n_components=2)
    #    #pca3D = PCA(n_components=3)

    #    comps = pca.fit_transform(group)

    #    pcas.append(comps)
    #    #comps3D = pca3D.fit_transform(group)

    comps2D = pca2D.fit_transform(Xtrain)
    comps3D = pca3D.fit_transform(Xtrain)
    comps = pca.fit_transform(Xtrain)

    u, s, v = np.linalg.svd(Xtrain, full_matrices=False)
    #np.savetxt('PCA.csv', comps, delimiter=',')
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    fig = plt.figure(figsize=(10,10))
    ax2D = fig.add_subplot(121)
    ax3D = fig.add_subplot(122, projection='3d')
    ax2D.set_xlabel('principal component 1')
    ax2D.set_ylabel('principal component 2')
    ax3D.set_xlabel('principal component 1')
    ax3D.set_ylabel('principal component 2')
    ax3D.set_zlabel('principal component 3')

    for i, sample in enumerate(comps2D):

        ax2D.scatter(sample[0], sample[1], c=[colors[L[i]]])

    for i, sample in enumerate(comps3D):

        ax3D.scatter(sample[0], sample[1], c=[colors[L[i]]])
    
    plt.show()

    #fig = plt.figure(figsize=(10,10))
    #ax2D = fig.add_subplot(121)
    #ax3D = fig.add_subplot(122, projection='3d')
    #ax2D.set_xlabel('principal component 1')
    #ax2D.set_ylabel('principal component 2')
    #ax3D.set_xlabel('principal component 1')
    #ax3D.set_ylabel('principal component 2')
    #ax3D.set_zlabel('principal component 3')

    #for i, sample in enumerate(pcas[2]):

    #    ax2D.scatter(sample[0], sample[1], c='r')

    #for i, sample in enumerate(pcas[1]):

    #    ax3D.scatter(sample[0], sample[1], c='r')
    
    #plt.show()


if __name__ == "__main__":
    main()
