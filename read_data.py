import scipy.sparse as sp
from KNN import knn
import GNN as gnn
import numpy as np
import json
#import ipdb



def use_knn(Xtrain, L, Xtest, K):
    """This function is basically a wrapper function that does some pre-processing before calling shitty K-NN model"""
    train_x = Xtrain[:200]
    train_y = L[:200]
    test_x = Xtrain[200:]
    test_y = L[200:]
    return knn(K, train_x, train_y, test_x, test_y)


def create_matrix(adj, feat):
    """This function creates the adjacency matrix and the data matrix, returning that"""

    # Create the adjacency matrix
    A = np.zeros(adj.shape)
    indices, ind_ptr = adj.indices, adj.indptr
    current = 0
    for i in range(1, len(ind_ptr)):
        num_neighbors = ind_ptr[i]
        neighbors = indices[current:num_neighbors]
        current = num_neighbors

        # Add the neighbors for a specific node
        for n in neighbors:
            A[i - 1][n] = 1


def main():
    adj = sp.load_npz('adj.npz')
    feat = np.load('features.npy')
    L = np.load('labels.npy')
    splits = json.load(open('splits.json'))
    idx_train, idx_test = splits['idx_train'], splits['idx_test']
    idx_train_s, idx_test_s = set(idx_train), set(idx_test)
    Xtrain, Xtest = np.zeros((len(idx_train), feat.shape[1])), np.zeros((len(idx_test), feat.shape[1]))
    adj_all = sp.csr_matrix.toarray(adj)

    # Populates the training and testing datasets
    for i, v in enumerate(idx_train):
        Xtrain[i, :] = feat[v]
    for i, v in enumerate(idx_test):
        Xtest[i, :] = feat[v]

    # Create the adjacency matrix and training/testing dataset
    # Atrain = [[1 if (c in idx_train_s and row[c]) else 0 for c in range(len(row))] for row in adj_all]
    # Atest = [[1 if (c in idx_test_s and row[c]) else 0 for c in range(len(row))] for row in adj_all]

    # Create new adjacency matrix for only the 496 x 496 training samples
    adj_train = sp.csr_matrix.toarray(adj)
    idx_to_remove = sorted(idx_test, reverse=True)
    for idx in idx_to_remove:
        adj_train = np.delete(adj_train, idx, 0)
        adj_train = np.delete(adj_train, idx, 1)

    # Create new adjacency matrix for only the 1984 x 1984 testing samples
    adj_test = sp.csr_matrix.toarray(adj)
    idx_to_remove = sorted(idx_train, reverse=True)
    for idx in idx_to_remove:
        adj_test = np.delete(adj_test, idx, 0)
        adj_test = np.delete(adj_test, idx, 1)

    # GNN stuff
    t = gnn.avg_neighbors(0, A, feat)

    # Call K-NN function
    K = int(input("Enter the number of K nearest neighbor: "))
    predictions = use_knn(Xtrain, L, Xtest, K)

    # Writing your prediction to submission.txt
    predictions = np.array(predictions).astype(np.int)
    np.savetxt('submission.txt', predictions, fmt='%d')
    # to manually look at the predictions and true labels
    #print(L[300:])
    #print(predictions)

    # to test the error for the
    #error = 0
    #for i in range(len(predictions)):
    #   if predictions[i] != L[i+300]:
    #       error+=1
    #print(error/len(predictions))


if __name__ == "__main__":
    main()

