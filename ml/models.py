import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import dgl
from dgl.nn import GraphConv



#Every PyTorch model must inherit from torch.nn.Module
#nn.Module has very useful functions for models
class Net(nn.Module):
    # Declaring the Architectur
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
 
    # Forward Pass
    def forward(self, x):
        x = x.view(x.shape[0],-1) # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        x = torch.FloatTensor(x)
        return self.forward(x)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_feat, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def predict(self,x, edge_index):
        return self.forward(x, edge_index)



class GCNdgl(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCNdgl, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
