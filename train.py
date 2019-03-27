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
from copy import deepcopy

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
import utils.evaluation
import utils.plotaccuracies
import utils.plotcontour
import utils.plotgraph
import utils.plotlosses

log = logging.getLogger(__name__)

NUM_WORKERS = 4

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

def generated_balanced_k_folds(dataset, indices, folds, randomSeed=32):

    log.info("Generating {0} balanced folds...".format(folds))

    log.info("Seed is {0}".format(randomSeed))

    fold_indices_ = []
    for i in range(folds):
        fold_indices_.append([])

    np.random.seed(randomSeed)
    np.random.shuffle(indices)
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

def train_kfolds(args, experimentStr, dataset, foldsIdx, datasetTest=None):

    log.info("Training with k={0} folds...".format(len(foldsIdx)))

    avg_train_accuracy_ = 0.0
    avg_validation_accuracy_ = 0.0
    avg_test_accuracy_ = 0.0

    per_fold_train_accuracies_ = []
    per_fold_losses_ = []
    per_fold_validation_accuracies_ = []
    per_fold_test_accuracies_ = []

    for fold in range(args.folds):

        log.info("Training with fold {0} left out...".format(fold))

        validation_fold_idx_ = foldsIdx[fold]
        train_fold_idx_ = foldsIdx.copy()
        train_fold_idx_.pop(fold)
        train_fold_idx_ = [item for sublist in train_fold_idx_ for item in sublist]

        train_sampler_ = SubsetRandomSampler(train_fold_idx_)
        validation_sampler_ = SubsetRandomSampler(validation_fold_idx_)

        train_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler_, num_workers=NUM_WORKERS)
        validation_loader_ = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=validation_sampler_, num_workers=NUM_WORKERS)

        test_loader_ = None
        if (args.test):
            test_loader_ = DataLoader(datasetTest, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        log.info("Batch size is {0}".format(args.batch_size))

        ## Select CUDA device
        device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        log.info(device_)
        log.info(torch.cuda.get_device_name(0))

        ## Build model
        model_ = network.utils.get_network(args.network, dataset.data.num_features, dataset.data.num_classes).to(device_)
        #model_ = torch.nn.DataParallel(model_, device_ids=range(torch.cuda.device_count()))
        log.info(model_)

        ## Optimizer
        optimizer_ = torch.optim.Adam(model_.parameters(), lr=args.lr, weight_decay=5e-4)
        log.info(optimizer_)

        ## Log accuracies, learning rate, and loss
        epochs_ = []
        train_accuracies_ = []
        train_losses_ = []
        validation_accuracies_ = []
        best_val_acc_ = 0.0
        test_accuracies_ = []
        best_test_acc_ = 0.0

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

                ff_noise_std = 287
                mf_noise_std = 197
                th_noise_std = 236

                ff_noise = np.random.randint(-ff_noise_std, high=ff_noise_std, size=batch.x[:, 0].shape)
                mf_noise = np.random.randint(-mf_noise_std, high=mf_noise_std, size=batch.x[:, 1].shape)
                th_noise = np.random.randint(-th_noise_std, high=th_noise_std, size=batch.x[:, 2].shape)
                noise = np.array([ff_noise, mf_noise, th_noise]).T

                augmented_batch = deepcopy(batch)
                augmented_batch.x = augmented_batch.x + torch.from_numpy(noise).type(torch.FloatTensor)

                samples = [batch, augmented_batch]

                for sample in samples:

                    sample = sample.to(device_)
                    optimizer_.zero_grad()
                    output_ = model_(sample)
                    loss_ = F.nll_loss(output_, sample.y)
                    loss_.backward()
                    loss_all += sample.y.size(0) * loss_.item()
                    optimizer_.step()

                i+=1

            # Log loss
            train_losses_.append(loss_all)
            log.info("Training loss {0}".format(loss_all))

            # Evaluate on training set
            model_.eval()
            correct_ = 0

            for batch in train_loader_:

                batch = batch.to(device_)
                pred_ = model_(batch).max(1)[1]
                correct_ += pred_.eq(batch.y).sum().item()

            correct_ /= len(train_fold_idx_)

            train_accuracies_.append(correct_)
            log.info("Training accuracy {0}".format(correct_))

            # Evaluate on validation set
            model_.eval()
            correct_ = 0

            for batch in validation_loader_:

                batch = batch.to(device_)
                pred_ = model_(batch).max(1)[1]
                correct_ += pred_.eq(batch.y).sum().item()

            correct_ /= len(validation_fold_idx_)

            validation_accuracies_.append(correct_)
            log.info("Validation accuracy {0}".format(correct_))

            # Checkpoint model if best validation accuracy found
            if correct_ > best_val_acc_ and args.save_ckpt:

                log.info("BEST VALIDATION ACCURACY SO FAR, checkpoint model...")

                best_val_acc_ = correct_

                state_ = {'epoch': epoch+1,
                      'model_state': model_.state_dict(),
                      'optimizer_state': optimizer_.state_dict(),}
                torch.save(state_, (args.ckpt_path + "/" + experimentStr + "_fold{0}_val.pkl").format(fold))

            # Evaluate on test set if required
            if (args.test):

                model_.eval()
                correct_ = 0

                for batch in test_loader_:

                    batch = batch.to(device_)
                    pred_ = model_(batch).max(1)[1]
                    correct_ += pred_.eq(batch.y).sum().item()

                correct_ /= len(datasetTest)

                test_accuracies_.append(correct_)
                log.info("Test accuracy {0}".format(correct_))

                utils.evaluation.eval(model_, device_, test_loader_, plot=False)

                # Checkpoint model if best test accuracy found
                if correct_ > best_test_acc_ and args.save_ckpt:
                    
                    log.info("BEST TEST ACCURACY SO FAR, checkpoint model...")

                    best_test_acc_ = correct_

                    state_ = {'epoch': epoch+1,
                          'model_state': model_.state_dict(),
                        'optimizer_state': optimizer_.state_dict(),}
                    torch.save(state_, (args.ckpt_path + "/" + experimentStr + "_fold{0}_test.pkl").format(fold))

            epochs_.append(epoch)

        time_end_ = timer()
        log.info("Training took {0} seconds".format(time_end_ - time_start_))

        max_accuracy_index_ = validation_accuracies_.index(max(validation_accuracies_))

        log.info("Maximum validation accuracy {0}".format(validation_accuracies_[max_accuracy_index_]))
        log.info("Training accuracy {0}".format(train_accuracies_[max_accuracy_index_]))
        log.info("At epoch {0}".format(max_accuracy_index_))

        if (args.test):
            max_test_accuracy_index_ = test_accuracies_.index(max(test_accuracies_))

            log.info("Maximum test accuracy {0}".format(test_accuracies_[max_test_accuracy_index_]))
            log.info("Validation accuracy {0}".format(validation_accuracies_[max_test_accuracy_index_]))
            log.info("Training accuracy {0}".format(train_accuracies_[max_test_accuracy_index_]))
            log.info("At epoch {0}".format(max_test_accuracy_index_))
            
            avg_test_accuracy_ += test_accuracies_[max_test_accuracy_index_]
            
            per_fold_test_accuracies_.append(test_accuracies_)
        
        avg_train_accuracy_ += train_accuracies_[max_accuracy_index_]
        avg_validation_accuracy_ += validation_accuracies_[max_accuracy_index_]

        per_fold_train_accuracies_.append(train_accuracies_)
        per_fold_losses_.append(train_losses_)
        per_fold_validation_accuracies_.append(validation_accuracies_)

    avg_train_accuracy_ /= args.folds
    avg_validation_accuracy_ /= args.folds

    log.info("Average training accuracy {0}".format(avg_train_accuracy_))
    log.info("Averate validation accuracy {0}".format(avg_validation_accuracy_))

    if (args.test):
        avg_test_accuracy_ /= args.folds

        log.info("Averate test accuracy {0}".format(avg_test_accuracy_))

    if (args.visualize_plots):

        epochs_ = [i for i in range(args.epochs)]
        labels_ = ["Fold {0}".format(i) for i in range(args.folds)]

        utils.plotaccuracies.plot_accuracies(epochs_, per_fold_train_accuracies_, labels_, "Train Accuracy")
        utils.plotaccuracies.plot_accuracies(epochs_, per_fold_validation_accuracies_, labels_, "Validation Accuracy")

        if (args.test):
            utils.plotaccuracies.plot_accuracies(epochs_, per_fold_test_accuracies_, labels_, "Test Accuracy")

        utils.plotlosses.plot_losses(epochs_, per_fold_losses_, labels_, "Train Loss")

def train(args, experimentStr):

    biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split="train", normalize=args.normalize)
    log.info("Training dataset...")
    log.info(biotacsp_dataset_)

    biotacsp_dataset_test_ = None

    if (args.test):
        biotacsp_dataset_test_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split="test", normalize=args.normalize)
        log.info("Testing dataset...")
        log.info(biotacsp_dataset_test_)

    # REMEMBER: Pass a random seed to generators when no need to replicate experiments
    random_seed_ = int(time.time())
    #random_seed_ = 32

    if (args.folds > 0):
        folds_idx_ = generated_balanced_k_folds(biotacsp_dataset_, list(range(len(biotacsp_dataset_))), args.folds, random_seed_)
        train_kfolds(args, experimentStr, biotacsp_dataset_, folds_idx_, biotacsp_dataset_test_)
    else:
        log.info("Folds must be greater than zero...")

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--log_path", nargs="?", default="logs", help="Logging path")
    parser_.add_argument("--ckpt_path", nargs="?", default="ckpts", help="Path to save checkpoints")
    parser_.add_argument("--save_ckpt", nargs="?", type=bool, default=True, help="Wether or not to store the best weights")
    parser_.add_argument("--normalize", nargs="?", type=bool, default=False, help="Normalize dataset using feature scaling")
    parser_.add_argument("--graph_k", nargs="?", type=int, default=0, help="K-Neighbours for graph connections, use 0 for manual connections")
    parser_.add_argument("--folds", nargs="?", type=int, default=5, help="Number of folds for k-fold cross validation, use 1 for no cross-validation")
    parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    parser_.add_argument("--network", nargs="?", default="GCN_test", help="The network model to train")
    parser_.add_argument("--lr", nargs="?", type=float, default=0.0001, help="Learning Rate")
    parser_.add_argument("--epochs", nargs="?", type=int, default=32, help="Training Epochs")
    parser_.add_argument("--test", nargs="?", type=bool, default=False, help="Enables testing while training")
    parser_.add_argument("--visualize_batch", nargs="?", type=bool, default=False, help="Wether or not to display batch contour plots")
    parser_.add_argument("--visualize_plots", nargs="?", type=bool, default=False, help="Enable visualization of learning plots")

    args_ = parser_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Experiment name (and log filename) follows the format network-normalization-graph_k-datetime
    experiment_str_ = "train-{0}-{1}-{2}-{3}".format(
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