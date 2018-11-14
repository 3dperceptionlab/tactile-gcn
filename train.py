__author__      = "Alberto Garcia-Garcia and Brayan Zapata-Impata"
__copyright__   = "Copyright 2018, 3D Perception Lab"
__credits__     = ["Alberto Garcia-Garcia",
                    "Brayan Zapata-Impata"]

__license__     = "MIT"
__version__     = "1.0"
__maintainer__  = "Alberto Garcia-Garcia"
__email__       = "agarcia@dtic.ua.es"
__status__ = "Development"

import argparse
import logging
import sys
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
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

def generate_balanced_train_test_splits(dataset, random_seed=32, split=0.2):

    log.info("Generating balanced train/test splits...")

    train_indices_ = []
    test_indices_ = []
    train_class_count_ = {}
    test_class_count_ = {}

    num_samples_ = len(dataset)
    indices_ = list(range(num_samples_))
    num_samples_test_ = int(np.floor(split * num_samples_))

    log.info("Dataset has {0} elements...".format(num_samples_))
    log.info("Split is {0}".format(split))
    log.info("Testing split will contain {0} samples".format(num_samples_test_))

    test_class_count_target_ = np.floor(dataset.class_weights * num_samples_test_)

    log.info("The target distribution for the test set is {0}".format(test_class_count_target_))

    np.random.seed(random_seed)
    np.random.shuffle(indices_)

    for i in range(len(indices_)):
        
        label_ = dataset[i].y.item()

        if test_class_count_target_[label_] > test_class_count_.get(label_, 0):
            test_indices_.append(indices_[i])
            test_class_count_[label_] = test_class_count_.get(label_, 0) + 1
        else:
            train_indices_.append(indices_[i])
            train_class_count_[label_] = train_class_count_.get(label_, 0) + 1

    log.info("Train split contains {0} samples and test split contain {1} samples...".format(len(train_indices_), len(test_indices_)))
    log.info("Class count on the training split is {0}".format(train_class_count_))
    log.info("Class count on the test split is {0}".format(test_class_count_))

    return train_indices_, test_indices_

def train(args):

    ## Dataset and loaders
    biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp')
    log.info(biotacsp_dataset_)

    train_idx_, test_idx_ = generate_balanced_train_test_splits(biotacsp_dataset_)
    train_sampler_ = SubsetRandomSampler(train_idx_)
    test_sampler_ = SubsetRandomSampler(test_idx_)

    biotacsp_train_loader_ = DataLoader(
        biotacsp_dataset_, batch_size=args.batch_size, shuffle=False, sampler=train_sampler_, num_workers=1)
    biotacsp_test_loader_ = DataLoader(
        biotacsp_dataset_, batch_size=args.batch_size, shuffle=False, sampler=test_sampler_, num_workers=1)

    log.info("Batch size is {0}".format(args.batch_size))

    ## Select CUDA device
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(device_)
    log.info(torch.cuda.get_device_name(0))

    ## Build model
    model_ = Net(biotacsp_dataset_.data.num_features, biotacsp_dataset_.data.num_classes).to(device_)
    log.info(model_)

    ## Optimizer
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
    log.info(optimizer_)

    time_start_ = timer()
    
    for epoch in range(args.epochs):

        log.info("Training epoch {0} out of {1}".format(epoch, args.epochs))

        model_.train()
        loss_all = 0

        i = 1
        for batch in biotacsp_train_loader_:

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

        for batch in biotacsp_train_loader_:

            batch = batch.to(device_)
            pred_ = model_(batch).max(1)[1]
            correct_ += pred_.eq(batch.y).sum().item()

        correct_ /= len(train_idx_)

        log.info("Training accuracy {0}".format(correct_))

        model_.eval()
        correct_ = 0

        for batch in biotacsp_test_loader_:

            batch = batch.to(device_)
            pred_ = model_(batch).max(1)[1]
            correct_ += pred_.eq(batch.y).sum().item()

        correct_ /= len(test_idx_)

        log.info("Test accuracy {0}".format(correct_))

    time_end_ = timer()
    log.info("Training took {0} seconds".format(time_end_ - time_start_))

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    parser_.add_argument("--lr", nargs="?", type=float, default=0.0001, help="Learning Rate")
    parser_.add_argument("--epochs", nargs="?", type=int, default=32, help="Training Epochs")

    args_ = parser_.parse_args()

    train(args_)
