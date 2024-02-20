import json
import numpy as np
import scipy.spatial
import scipy.sparse
from scipy.io import savemat
from cross_validation import cross_val
from sklearn.metrics import confusion_matrix

f = open('splits.json')

datasplit = json.load(f)

labels = np.load('labels.npy')
features = np.load('features.npy')
adj = scipy.sparse.load_npz('adj.npz')
# data = np.load('adj/data.npy')
# format = np.load('adj/format.npy')
# indices = np.load('adj/indices.npy')
# indptr = np.load('adj/indptr.npy')
# shape = np.load('adj/shape.npy')

trainfeatures = []

for i in datasplit["idx_train"]:
    trainfeatures.append(features[i])

testfeatures = []

for i in datasplit["idx_test"]:
    testfeatures.append(features[i])


# print(adj.shape)
# for i in data['idx_train']:
#     print(i)
# print(datasplit)
# print(type(adj))
# adj = scipy.sparse.csr_matrix.toarray(adj)
# print(adj.shape)
# print(sum(adj).tolist())
# print(adj[0])
# for i in range(10):
#     print(adj.indices[i])

adj = scipy.sparse.csr_matrix.toarray(adj)

# print(adj[0][1083])
# print(adj[0][1084])
# print(adj[0][1085])
# for i in adj:
#     print(i)
#     print()

# savemat("features.mat", features)

# savemat("labels.mat", labels)
# savemat("train.mat", datasplit["idx_train"])
# savemat("test.mat", datasplit["idx_test"])



# # savemat("adj.mat", adj)
# np.savetxt('trainfeatures.csv', trainfeatures, delimiter=',')
# np.savetxt('testfeatures.csv', testfeatures, delimiter=',')
# np.savetxt('labels.csv', labels, delimiter=',') 
# np.savetxt('train.csv', datasplit["idx_train"], delimiter=',') 
# np.savetxt('test.csv', datasplit["idx_train"], delimiter=',') 
# np.savetxt('adj.csv', adj, delimiter=',') 



# numpy.savetxt("foo.csv", a, delimiter=",")

# for item in adj:
    # print(item)

# for key, value in adj.items():


# traintrainfeatures = trainfeatures[200:]
# traintestfeatures = trainfeatures[:200]
#
# trainlabels = labels[200:]
# testlabels = labels[:200]

groups = []

### here we did our sanity check to make sure the results we are getting are not crazy. so we are doing cross validation on the training features

'''
K = int(input("Enter the number of folds: "))
xtrain_all, xtest_all, ytrain_all, ytest_all = cross_val(K, trainfeatures, labels)
error_all = 0

# Goes through each instance of our folds dataset
for inst in range(K):

    # Gets the specific training and testing sets for a fold
    traintrainfeatures = xtrain_all[inst]
    traintestfeatures = xtest_all[inst]
    trainlabels = ytrain_all[inst]
    testlabels = ytest_all[inst]

    # Bins the features
    for i in range(7):
        local = []
        for j in range(len(labels)):
            if i == labels[j]:
                local.append(trainfeatures[j])
        groups.append(local)

    # Gets the average for each class
    groupsaverage = []
    for i in range(7):
        groupsaverage.append(np.mean(groups[i], axis=0))

    # Predicts the label for a test node
    pred = []
    for i in traintestfeatures:
        local = []
        for j in groupsaverage:
            local.append(scipy.spatial.distance.euclidean(i, j))
        pred.append(local.index(min(local)))

    # Calculate the error
    error = 0
    for i in range(len(pred)):
        if pred[i] != testlabels[i]:
            error += 1
            error_all += 1
    print("Fold", str(inst) + ":", str(100 * round(error/len(pred), 3)) + "%")

    # Does confusion matrix shit
    cm = confusion_matrix(pred, testlabels)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) \
             if i != j]) / len(testlabels)
    print("\nK={}, error={}\n".format(inst, round(w, 3)))
    print(cm, end="\n\n")

print("Average Error Between Folds:", str(100 * round(error_all/(len(trainfeatures)), 3)) + "%")
'''



### find the the test node labels based on closest centroid


for i in range(7):
    local = []
    for j in range(len(labels)):
        if i == labels[j]:
            local.append(trainfeatures[j])
    groups.append(local)

groupsaverage = []
for i in range(7):
    groupsaverage.append(np.mean(groups[i], axis=0))


pred = []
for i in testfeatures:
    local = []
    for j in groupsaverage:
        local.append(scipy.spatial.distance.euclidean(i, j))
    pred.append(local.index(min(local)))

print(pred)

predictions = np.array(pred).astype(np.int)
np.savetxt('submission.txt', pred, fmt='%d')