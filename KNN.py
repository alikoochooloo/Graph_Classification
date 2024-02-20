from operator import itemgetter
import scipy.spatial.distance as sp
from sklearn.metrics import confusion_matrix


def classify_knn(k, distances):
    """
    This function takes in the k (int) as the number of nearest neighbors and
    the distances (list[float]) from a test point to all the training points. 
    The closest K neighbors are assessed, and the most prevelant class is the
    one the test point will be assigned to
    """
    if k == 1:
        distances_k = [min(distances, key=itemgetter(0))]
    else:
        distances_k = sorted(distances, key=itemgetter(0))[:k]
    counts = [0, 0, 0, 0, 0, 0, 0]
    for d in distances_k:
        counts[d[1]] += 1
    return max(enumerate(counts), key=itemgetter(1))[0]


def knn(k, train_x, train_y, test_x, test_y):
    """
    This function takes in the k (int) as the number of nearest neighbors and
    all the training and testing input and ground truth lists. It loops over
    all the testing points, and for each testing point, if finds its distance
    to all of training points. It calculates the error rate for classification
    """
    predicted = []
    for test_i, test_sample in enumerate(test_x):
        distances = []
        for train_i, train_sample in enumerate(train_x):
            distances.append((sp.euclidean(test_sample, train_sample),
                              train_y[train_i]))
        predicted.append(classify_knn(k, distances))

    cm = confusion_matrix(predicted, test_y)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) \
            if i != j]) / len(test_y)
    print("\nK={}, error={}\n".format(k, round(w, 3)))
    print(cm, end="\n\n")
    return predicted