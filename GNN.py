from matplotlib.style import use
import scipy.sparse as sp
import numpy as np
import scipy.spatial
import scipy.sparse
import json
from sklearn.utils import shuffle
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import torch.optim as optim


class GNN(nn.Module):

    def __init__(self):
        super(GNN, self).__init__()
        self.fc = nn.Linear(2780, 500)
        self.fc1 = nn.Linear(500, 150)  # 6*6 from image dimension
        self.fc2 = nn.Linear(150, 30)
        self.fc3 = nn.Linear(30, 7)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def avg_neighbors(adj, data):
    # neighbors = [i for i, col in enumerate(adj[node]) if col == 1]

    # featAvg = np.zeros(1390)

    # for neighbor in neighbors:
    #     featAvg += 1/len(neighbors) * data[neighbor]

    count = sum(adj)

    featAvg = np.divide(np.dot(adj,data),count)

    return featAvg

def use_averages(features, adj, nodes):
    averages = []

    for i in nodes:
        localadj = adj[i]
        averages.append(avg_neighbors(localadj, features))


    return np.hstack((features,averages))


if __name__ == '__main__':

    f = open('splits.json')

    datasplit = json.load(f)

    labels = np.load('labels.npy')
    features = np.load('features.npy')
    adj = scipy.sparse.load_npz('adj.npz')
    adj = scipy.sparse.csr_matrix.toarray(adj)
    features = use_averages(features,adj,range(2480))
    # labels = labels.type(torch.LongTensor)
    # templabels = []
    # for i in range(len(labels)):
    #     temp = [0,0,0,0,0,0,0]
    #     temp[labels[i]] = 1
    #     templabels.append(temp)
    # labels = np.array(templabels)

    # print(labels)
    # print(features.shape)

    trainfeatures = []

    for i in datasplit["idx_train"]:
        trainfeatures.append(features[i])
    
    # features = []
    # for (i , j) in zip(trainfeatures, labels):
    #     features.append([i,j])
    train = trainfeatures[:300]
    train_l = labels[:300]
    test = trainfeatures[300:]
    test_l = labels[300:]

    # print(train[0][0].shape)
    model = GNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)

    train = torch.utils.data.DataLoader(train, batch_size = 3, shuffle = False, num_workers = 2)
    train_l = torch.utils.data.DataLoader(train_l, batch_size = 3, shuffle = False, num_workers = 2)

    test = torch.utils.data.DataLoader(test, batch_size = 3, shuffle = False, num_workers = 2)
    test_l = torch.utils.data.DataLoader(test_l, batch_size = 3, shuffle = False, num_workers = 2)

    # print(iter(train_l).next())
    

    all_loss = []
    for epoch in range(30):
        temp_loss = []
        for (f, l) in zip(iter(train),iter(train_l)):

            output = model(f.float())
            # print(output)
            # print(l.type(torch.LongTensor))

            loss = criterion(output, l.type(torch.LongTensor))
            temp_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch: {epoch}, loss: {np.mean(temp_loss)}")
    
    correct, total = 0, 0

    with torch.no_grad():

        for (f, l) in zip(iter(test),iter(test_l)):
            output = model(f.float())

            _, predicted = torch.max(output.data, 1)
            # print(predicted)
            # print(l)
            # print((predicted == l).sum().item())

            total += l.size(0)
            correct += (predicted == l).sum().item()

    print(100 * correct / total)

