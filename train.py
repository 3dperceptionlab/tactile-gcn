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
import time
from timeit import default_timer as timer

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

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
import utils.plotconfusionmatrix
import utils.plotaccuracies
import utils.plotcontour
import utils.plotgraph
import utils.plotlosses

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

def generate_balanced_train_test_splits(dataset, randomSeed=32, split=0.2):

    log.info("Generating balanced train/test splits...")

    log.info("Seed is {0}".format(randomSeed))

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

    np.random.seed(randomSeed)
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

def generated_balanced_k_folds(dataset, indices, folds, randomSeed=32):

    log.info("Generating {0} balanced folds...".format(folds))

    log.info("Seed is {0}".format(randomSeed))

    fold_indices_ = []
    for i in range(folds):
        fold_indices_.append([])

    np.random.seed(randomSeed)
    np.random.shuffle(indices)
    log.info(indices)
    num_samples_ = len(indices)

    log.info("There are {0} indices to be distributed across the folds...".format(num_samples_))

    class_indices_ = np.zeros(dataset.num_classes, dtype=np.int)
    for i in range(num_samples_):
        label_ = dataset[int(indices[i])].y.item()
        fold_indices_[class_indices_[label_]].append(indices[i])
        class_indices_[label_] = (class_indices_[label_] + 1) % folds

    # Check length and class balance
    for i in range(folds):

        class_count_ = np.zeros(dataset.num_classes, dtype=int)
        num_samples_fold_ = len(fold_indices_[i])

        log.info("Fold {0} contains {1} samples...".format(i, num_samples_fold_))

        for k in range(num_samples_fold_):

            label_ = dataset[int(fold_indices_[i][k])].y.item()
            class_count_[label_] += 1

        log.info("Class count on that fold is {0}".format(class_count_))
        

    return fold_indices_

def evaluate(model, device, loader):

    ## Launch predictions on test and calculate metrics
    test_acc_ = 0.0
    test_y_ = []
    test_pred_ = []

    model.eval()

    for batch in loader:

        batch = batch.to(device)
        pred_ = model(batch).max(1)[1]
        test_acc_ += pred_.eq(batch.y).sum().item()

        test_y_.append(batch.y)
        test_pred_.append(pred_)

    # TODO: OJO A QUE ESTO NO COJA LA LONGITUD EN BATCHES
    log.info("CHECK CHECK CHECK: {0}".format(len(loader)))
    test_acc_ /= len(loader)

    test_prec_, test_rec_, test_fscore_, _ = precision_recall_fscore_support(test_y_, test_pred_, average='binary')

    log.info("Metrics")
    log.info("Accuracy: {0}".format(test_acc_))
    log.info("Precision: {0}".format(test_prec_))
    log.info("Recall: {0}".format(test_rec_))
    log.info("F-score: {0}".format(test_fscore_))

    conf_matrix_ = confusion_matrix(test_y_, test_pred_)

    ## Plot non-normalized confusion matrix
    utils.plotconfusionmatrix.plot_confusion_matrix(conf_matrix_, classes=np.unique(test_y_),
                        title='Confusion matrix, without normalization')

def train_traintest(args, dataset, trainIdx, testIdx):

    log.info("Trainign with train and test set...")

    train_sampler_ = SubsetRandomSampler(trainIdx)
    train_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler_, num_workers=1)

    test_sampler_ = SubsetRandomSampler(testIdx)
    test_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler_, num_workers=1)

    ## Select CUDA device
    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log.info(device_)
    log.info(torch.cuda.get_device_name(0))

    ## Build model
    model_ = Net(dataset.data.num_features, dataset.data.num_classes).to(device_)
    log.info(model_)

    ## Optimizer
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
    log.info(optimizer_)

    ## Log accuracies, learning rate, and loss
    epochs_ = []
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
            #log.info("Training batch {0} of {1}".format(i, len(dataset)/args.batch_size))
            #visualize_batch(batch)

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

        correct_ /= len(trainIdx)

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

        correct_ /= len(testIdx)
        test_accuracies_.append(correct_)

        log.info("Test accuracy {0}".format(correct_))

        epochs_.append(epoch)

    time_end_ = timer()
    log.info("Training took {0} seconds".format(time_end_ - time_start_))

    utils.plotaccuracies.plot_accuracies(epochs_, [train_accuracies_, test_accuracies_], ["Train Accuracy", "Test Accuracy"])
    utils.plotlosses.plot_losses(epochs_, [train_losses_], ["Train Loss"])

    evaluate(model_, device_, test_loader_)

def train_kfolds(args, dataset, foldsIdx):

    log.info("Training with k={0} folds...".format(len(foldsIdx)))

    avg_train_accuracy_ = 0.0
    avg_validation_accuracy_ = 0.0

    for fold in range(args.folds):

        log.info("Training with fold {0} left out...".format(fold))

        validation_fold_idx_ = foldsIdx[fold]
        train_fold_idx_ = foldsIdx.copy()
        train_fold_idx_.pop(fold)
        train_fold_idx_ = [item for sublist in train_fold_idx_ for item in sublist]

        train_sampler_ = SubsetRandomSampler(train_fold_idx_)
        validation_sampler_ = SubsetRandomSampler(validation_fold_idx_)

        train_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler_, num_workers=1)
        validation_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=validation_sampler_, num_workers=1)

        log.info("Batch size is {0}".format(args.batch_size))

        ## Select CUDA device
        device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log.info(device_)
        log.info(torch.cuda.get_device_name(0))

        ## Build model
        model_ = Net(dataset.data.num_features, dataset.data.num_classes).to(device_)
        log.info(model_)

        ## Optimizer
        optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
        log.info(optimizer_)

        ## Log accuracies, learning rate, and loss
        epochs_ = []
        train_accuracies_ = []
        validation_accuracies_ = []
        train_losses_ = []

        time_start_ = timer()
        
        for epoch in range(args.epochs):

            log.info("Training epoch {0} out of {1}".format(epoch, args.epochs))

            model_.train()
            loss_all = 0

            i = 1
            for batch in train_loader_:

                # Batch Visualization
                #log.info("Training batch {0} of {1}".format(i, len(dataset)/args.batch_size))
                #visualize_batch(batch)

                batch = batch.to(device_)
                optimizer_.zero_grad()
                output_ = model_(batch)
                loss_ = F.nll_loss(output_, batch.y)
                loss_.backward()
                loss_all += batch.y.size(0) * loss_.item()
                optimizer_.step()

                i+=1

            # Log loss
            train_losses_.append(loss_all)
            log.info("Training loss {0}".format(loss_all))

            model_.eval()
            correct_ = 0

            for batch in train_loader_:

                batch = batch.to(device_)
                pred_ = model_(batch).max(1)[1]
                correct_ += pred_.eq(batch.y).sum().item()

            correct_ /= len(train_fold_idx_)

            # Log train accuracy
            train_accuracies_.append(correct_)
            log.info("Training accuracy {0}".format(correct_))

            model_.eval()
            correct_ = 0

            for batch in validation_loader_:

                batch = batch.to(device_)
                pred_ = model_(batch).max(1)[1]
                correct_ += pred_.eq(batch.y).sum().item()

            correct_ /= len(validation_fold_idx_)
            validation_accuracies_.append(correct_)

            log.info("Validation accuracy {0}".format(correct_))

            epochs_.append(epoch)

        time_end_ = timer()
        log.info("Training took {0} seconds".format(time_end_ - time_start_))

        #utils.plotaccuracies.plot_accuracies(epochs_, [train_accuracies_, validation_accuracies_], ["Train Accuracy", "Test Accuracy"])
        #utils.plotlosses.plot_losses(epochs_, [train_losses_], ["Train Loss"])

        max_accuracy_index_ = validation_accuracies_.index(max(validation_accuracies_))
        avg_train_accuracy_ += train_accuracies_[max_accuracy_index_]
        avg_validation_accuracy_ += validation_accuracies_[max_accuracy_index_]

    avg_train_accuracy_ /= args.folds
    avg_validation_accuracy_ /= args.folds

    log.info("Average training accuracy {0}".format(avg_train_accuracy_))
    log.info("Averate validation accuracy {0}".format(avg_validation_accuracy_))

def train(args):

    biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k)
    log.info(biotacsp_dataset_)

    # REMEMBER: Pass a random seed to generators when no need to replicate experiments
    random_seed_ = int(time.time())
    #random_seed_ = 32

    if (args.folds > 1):
        folds_idx_ = generated_balanced_k_folds(biotacsp_dataset_, list(range(len(biotacsp_dataset_))), args.folds, random_seed_)
        train_kfolds(args, biotacsp_dataset_, folds_idx_)
    else:
        train_idx_, test_idx_ = generate_balanced_train_test_splits(biotacsp_dataset_, random_seed_)
        train_traintest(args, biotacsp_dataset_, train_idx_, test_idx_)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--graph_k", nargs="?", type=int, default=0, help="K-Neighbours for graph connections, use 0 for manual connections")
    parser_.add_argument("--folds", nargs="?", type=int, default=5, help="Number of folds for k-fold cross validation, use 1 for no cross-validation")
    parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    parser_.add_argument("--lr", nargs="?", type=float, default=0.0001, help="Learning Rate")
    parser_.add_argument("--epochs", nargs="?", type=int, default=32, help="Training Epochs")

    args_ = parser_.parse_args()

    train(args_)
