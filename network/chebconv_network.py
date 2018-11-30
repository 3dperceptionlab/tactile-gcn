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

class ChebConv_test(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = ChebConv(numFeatures, 8, 3)
        self.conv2 = ChebConv(8, 8, 3)
        self.fc1 = torch.nn.Linear(192, 64)
        self.fc2 = torch.nn.Linear(64, numClasses * 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))

        log.debug(data.x.view(-1).size())
        data.x = self.fc1(data.x.view(-1))
        data.x = self.fc2(data.x)
        log.debug(data.x.size())
        data.x = F.log_softmax(data.x.view(1, 2), dim=1)
        log.debug(data.x.size())
        return data.x


class ChebConv_8_16_32(torch.nn.Module):

    def __init__(self, numFeatures, numClasses):

        super().__init__()

        self.conv1 = ChebConv(numFeatures, 8, 3)
        self.conv2 = ChebConv(8, 16, 3)
        self.conv3 = ChebConv(16, 32, 5)
        self.fc1 = torch.nn.Linear(768, 128)
        self.fc2 = torch.nn.Linear(128, numClasses * 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index))
        data.x = F.elu(self.conv2(data.x, data.edge_index))
        data.x = F.elu(self.conv3(data.x, data.edge_index))

        log.debug(data.x.view(-1).size())
        data.x = self.fc1(data.x.view(-1))
        data.x = self.fc2(data.x)
        log.debug(data.x.size())
        data.x = F.log_softmax(data.x.view(1, 2), dim=1)
        log.debug(data.x.size())
        return data.x