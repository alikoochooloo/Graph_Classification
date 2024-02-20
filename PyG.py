import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    """Parameters
    ----
    num_node_features: input feature dimension for the GCN
    num_hidden: hidden dimension of the GCN
    num_classes: output dimension of the GCN
    """
    
    def __init__(self, num_node_features, num_hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden) 
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, edge_index):
        """
        ---
        x: node features
        edge_index: edges for the graph (could be converted from the adjacency matrix)
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x) # activation function
        x = F.dropout(x, training=self.training) # dropout to allievate overfitting
        x = self.conv2(x, edge_index) 

        return F.log_softmax(x, dim=1) # softmax to construct predicted class distribution

adj = sp.load_npz('adj.npz')
feat  = np.load('features.npy')
labels = np.load('labels.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from torch_geometric.utils import from_scipy_sparse_matrix
our_edge_index, _ = from_scipy_sparse_matrix(adj)

our_x = torch.FloatTensor(feat)



our_model = GCN(num_node_features=our_x.shape[1], 
            num_hidden=36,
            num_classes=labels.max()+1,
           ).to(device)

optimizer = torch.optim.Adam(our_model.parameters(), lr=0.01, weight_decay=5e-4)

our_model.train()

for epoch in range(200): # write a loop to train the model
    optimizer.zero_grad() # we need to eliminate the grads in the parameters
    out = our_model(our_x, our_edge_index) # forward the inputs to the GCN model
    loss = F.nll_loss(out, labels) # calculate the loss
    loss.backward() # to do backprogation; calculate the gradients for the parameters
    optimizer.step() # to update the parameters based on the calculated gradients
    if epoch % 10 == 0:
        print('Epoch {0}: {1}'.format(epoch, loss.item()))

# our_model(our_x, our_edge_index)