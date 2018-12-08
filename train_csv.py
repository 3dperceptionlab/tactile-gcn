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
import datetime
import logging
import sys
import time
from timeit import default_timer as timer

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import loader.biotacsp_loader
import dataset.biotacsp
import network.utils
import transforms.tograph
import utils.plotaccuracies
import utils.plotcontour
import utils.plotgraph
import utils.plotlosses

log = logging.getLogger(__name__)

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

        utils.plotgraph.plot_graph_3d(pos_, x_, y_, edge_index_)
        utils.plotgraph.plot_contourgraph_batch(pos_, x_, y_, edge_index_)

def traintest(args, experimentStr, datasetTrain, datasetTest):

    log.info("Training and testing...")

    train_loader_ = DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader_ = DataLoader(datasetTest, batch_size=args.batch_size, shuffle=False, num_workers=1)

    ## Select CUDA device
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(device_)
    log.info(torch.cuda.get_device_name(0))

    ## Build model
    model_ = network.utils.get_network(args.network, datasetTrain.data.num_features, datasetTrain.data.num_classes).to(device_)
    #model_ = torch.nn.DataParallel(model_, device_ids=range(torch.cuda.device_count()))
    log.info(model_)

    ## Optimizer
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
    log.info(optimizer_)

    ## Log accuracies, learning rate, and loss
    epochs_ = []
    best_test_acc_ = 0.0
    train_accuracies_ = []
    test_accuracies_ = []
    train_losses_ = []

    time_start_ = timer()
    
    for epoch in range(args.epochs):

        log.info("Training epoch {0} out of {1}".format(epoch, args.epochs))

        model_.train()
        loss_all = 0

        i = 1
        for batch in train_loader_:

            # Batch Visualization
            if (args.visualize_batch):
                log.info("Training batch {0} of {1}".format(i, len(dataset)/args.batch_size))
                visualize_batch(batch)

            batch = batch.to(device_)
            optimizer_.zero_grad()
            output_ = model_(batch)
            loss_ = F.nll_loss(output_, batch.y)
            loss_.backward()
            loss_all += batch.y.size(0) * loss_.item()
            optimizer_.step()

            i+=1

        # Log train loss
        train_losses_.append(loss_all)
        log.info("Training loss {0}".format(loss_all))

        # Get train accuracy
        model_.eval()
        correct_ = 0

        for batch in train_loader_:

            batch = batch.to(device_)
            pred_ = model_(batch).max(1)[1]
            correct_ += pred_.eq(batch.y).sum().item()

        correct_ /= len(datasetTrain)

        # Log train accuracy
        train_accuracies_.append(correct_)
        log.info("Training accuracy {0}".format(correct_))

        # Get test accuracy
        model_.eval()
        correct_ = 0

        for batch in test_loader_:

            batch = batch.to(device_)
            pred_ = model_(batch).max(1)[1]
            correct_ += pred_.eq(batch.y).sum().item()

        correct_ /= len(datasetTest)

        # Log test accuracy
        test_accuracies_.append(correct_)
        log.info("Test accuracy {0}".format(correct_))

        # Checkpoint model
        if correct_ > best_test_acc_ and args.save_ckpt:

            log.info("BEST ACCURACY SO FAR, checkpoint model...")

            best_test_acc_ = correct_

            state_ = {'epoch': epoch+1,
                      'model_state': model_.state_dict(),
                      'optimizer_state': optimizer_.state_dict(),}
            torch.save(state_, (args.ckpt_path + "/" + experimentStr + "_{0}.pkl").format(epoch))

        epochs_.append(epoch)

    time_end_ = timer()
    log.info("Training took {0} seconds".format(time_end_ - time_start_))

    utils.plotaccuracies.plot_accuracies(epochs_, [train_accuracies_, test_accuracies_], ["Train Accuracy", "Test Accuracy"])
    utils.plotlosses.plot_losses(epochs_, [train_losses_], ["Train Loss"])

def train(args, experimentStr):

    biotacsp_dataset_train_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split=None, csvs=args.train_csvs, normalize=args.normalize)
    biotacsp_dataset_test_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split=None, csvs=args.test_csvs, normalize=args.normalize)

    log.info(biotacsp_dataset_train_)
    log.info(biotacsp_dataset_test_)
    traintest(args, experimentStr, biotacsp_dataset_train_, biotacsp_dataset_test_)

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--train_csvs", nargs="+", help="<Required> Train CSV list", required=True)
    parser_.add_argument("--test_csvs", nargs="+", help="<Required> Test CSV list", required=True)
    parser_.add_argument("--log_path", nargs="?", default="logs", help="Logging path")
    parser_.add_argument("--ckpt_path", nargs="?", default="ckpts", help="Path to save checkpoints")
    parser_.add_argument("--save_ckpt", nargs="?", type=bool, default=True, help="Wether or not to store the best weights")
    parser_.add_argument("--normalize", nargs="?", type=bool, default=False, help="Normalize dataset using feature scaling")
    parser_.add_argument("--graph_k", nargs="?", type=int, default=0, help="K-Neighbours for graph connections, use 0 for manual connections")
    parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    parser_.add_argument("--network", nargs="?", default="GCN_test", help="The network model to train")
    parser_.add_argument("--lr", nargs="?", type=float, default=0.0001, help="Learning Rate")
    parser_.add_argument("--epochs", nargs="?", type=int, default=32, help="Training Epochs")
    parser_.add_argument("--visualize_batch", nargs="?", type=bool, default=False, help="Wether or not to display batch contour plots")

    args_ = parser_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Experiment name (and log filename) follows the format network-normalization-graph_k-datetime
    experiment_str_ = "traintest-{0}-{1}-{2}-{3}-{4}-{5}".format(
                        ''.join(args_.train_csvs),
                        ''.join(args_.test_csvs),
                        args_.network,
                        args_.normalize,
                        args_.graph_k,
                        datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

    # Add file handler to logging system to simultaneously log information to console and file
    log_formatter_ = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    file_handler_ = logging.FileHandler("{0}/{1}.log".format(args_.log_path, experiment_str_))
    file_handler_.setFormatter(log_formatter_)
    log.addHandler(file_handler_)

    train(args_, experiment_str_)
