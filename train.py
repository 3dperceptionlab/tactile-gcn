import logging
import sys

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv # noqa
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x,
global_mean_pool)

import loader.biotacsp_loader
import dataset.biotacsp
import transforms.tograph
import utils.plotcontour
import utils.plotgraph

log = logging.getLogger(__name__)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class Net(torch.nn.Module):
    def __init__(self, numFeatures, numClasses):
        super(Net, self).__init__()
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

def visualize_batch(batch):

    log.info(batch)
    log.info("Batch size {}".format(batch.num_graphs))

    npg_ = int(batch.num_nodes / batch.num_graphs)
    epg_ = int(batch.num_edges / batch.num_graphs)

    log.info(npg_)
    log.info(epg_)

    for i in range(batch.num_graphs):
        pos_ = batch['pos'][i*npg_:i*npg_ + npg_, :]
        x_ = batch['x'][i*npg_:i*npg_ + npg_, :]
        y_ = batch['y'][i:i]
        edge_index_ = batch['edge_index'][:, epg_*i:epg_*i + epg_] - i*npg_

        utils.plotgraph.plot_contourgraph_batch(pos_, x_, y_, edge_index_)


def train():

    BATCH_SIZE = 1

    biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp')
    biotacsp_loader_ = DataLoader(
        biotacsp_dataset_, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    log.info(biotacsp_dataset_)

    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(device_)
    log.info(torch.cuda.get_device_name(0))

    model_ = Net(biotacsp_dataset_.data.num_features, biotacsp_dataset_.data.num_classes).to(device_)
    log.info(model_)

    optimizer_ = torch.optim.Adam(model_.parameters(), lr=0.0001, weight_decay=5e-4)
    log.info(optimizer_)
    
    for epoch in range(32):

        log.info("Training epoch {0}".format(epoch))

        model_.train()
        loss_all = 0

        i = 1
        for batch in biotacsp_loader_:

            #log.info("Training batch {0} of {1}".format(i, len(biotacsp_dataset_)/BATCH_SIZE))

            batch = batch.to(device_)
            optimizer_.zero_grad()
            output_ = model_(batch)
            loss_ = F.nll_loss(output_, batch.y)
            loss_.backward()
            loss_all += batch.y.size(0) * loss_.item()
            optimizer_.step()

            i+=1

        model_.eval()
        correct_ = 0

        for batch in biotacsp_loader_:

            batch = batch.to(device_)
            pred_ = model_(batch).max(1)[1]
            correct_ += pred_.eq(batch.y).sum().item()

        correct_ /= len(biotacsp_dataset_)

        log.info("Training accuracy {0}".format(correct_))

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train()