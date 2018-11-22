import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from torch_geometric.nn import GCNConv, ChebConv # noqa
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x,
global_mean_pool)

log = logging.getLogger(__name__)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class GCN_test(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = GCNConv(numFeatures, 8)
        self.conv2 = GCNConv(8, 8)
        self.conv3 = GCNConv(8, 16)
        self.conv4 = GCNConv(16, 16)
        self.fc1 = torch.nn.Linear(384, 128)
        self.fc2 = torch.nn.Linear(128, numClasses * 1)

    def forward(self, data):
        #log.info("NETWOROOOOOORK")
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        data.x = F.relu(self.conv2(data.x, data.edge_index))
        data.x = F.relu(self.conv3(data.x, data.edge_index))
        data.x = F.relu(self.conv4(data.x, data.edge_index))
        #data.x = F.dropout(data.x, training=self.training)

        log.debug(data.x.view(-1).size())
        data.x = self.fc1(data.x.view(-1))
        data.x = self.fc2(data.x)
        log.debug(data.x.size())
        data.x = F.log_softmax(data.x.view(1, 2), dim=1)
        log.debug(data.x.size())
        return data.x

class GCN_32(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = GCNConv(numFeatures, 32)
        self.fc1 = torch.nn.Linear(32, numClasses)

    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        data.x = F.dropout(data.x, training=self.training)

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.x, batch = max_pool_x(cluster, data.x, data.batch)
        data.x = global_mean_pool(data.x, batch)

        data.x = self.fc1(data.x)
        data.x = F.log_softmax(data.x, dim=1)
        return data.x

class GCN_32_64(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = GCNConv(numFeatures, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc1 = torch.nn.Linear(64, numClasses)

    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        data.x = F.dropout(data.x, training=self.training)
        data.x = self.conv2(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.x, batch = max_pool_x(cluster, data.x, data.batch)
        data.x = global_mean_pool(data.x, batch)

        data.x = self.fc1(data.x)
        data.x = F.log_softmax(data.x, dim=1)
        return data.x

class GCN_32_64_128(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = GCNConv(numFeatures, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.fc1 = torch.nn.Linear(128, numClasses)

    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        data.x = F.dropout(data.x, training=self.training)
        data.x = self.conv2(data.x, data.edge_index)
        data.x = self.conv3(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.x, batch = max_pool_x(cluster, data.x, data.batch)
        data.x = global_mean_pool(data.x, batch)

        data.x = self.fc1(data.x)
        data.x = F.log_softmax(data.x, dim=1)
        return data.x
