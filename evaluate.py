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
import os
import sys
import time
from timeit import default_timer as timer

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import loader.biotacsp_loader
import dataset.biotacsp
import network.utils
import utils.evaluation

log = logging.getLogger(__name__)

def evaluate (args):

  dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split=args.split, normalize=args.normalize)
  log.info(dataset_)

  log.info("Evaluating network over {0} split...".format(args.split))

  eval_loader_ = DataLoader(dataset_, batch_size=args.batch_size, shuffle=False, num_workers=1)

  ## Select CUDA device
  device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  log.info(device_)
  log.info(torch.cuda.get_device_name(0))

  ## Build model
  model_ = network.utils.get_network(args.network, dataset_.data.num_features, dataset_.data.num_classes).to(device_)
  log.info(model_)

  # Load checkpoint if specified
  if args.checkpoint is not None:

    if os.path.isfile(args.checkpoint):

      log.info('Loading checkpoint {}'.format(args.checkpoint))
      checkpoint_ = torch.load(args.checkpoint)
      model_.load_state_dict(checkpoint_['model_state'])
      log.info('Loaded network...')

    else:
      log.info('The checkpoint file at {} was not found'.format(args.checkpoint))

  utils.evaluation.eval(model_, device_, eval_loader_)

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--log_path", nargs="?", default="logs", help="Logging path")
    parser_.add_argument("--split", nargs="?", default="test", help="Dataset split to evaluate")
    parser_.add_argument("--checkpoint", nargs="?", required=True, help="Path to save checkpoints")
    parser_.add_argument("--normalize", nargs="?", type=bool, default=True, help="Normalize dataset using feature scaling")
    parser_.add_argument("--graph_k", nargs="?", type=int, default=0, help="K-Neighbours for graph connections, use 0 for manual connections")
    parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
    parser_.add_argument("--network", nargs="?", default="GCN_test", help="The network model to train")

    args_ = parser_.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Experiment name (and log filename) follows the format network-normalization-graph_k-datetime
    experiment_str_ = "eval-{0}-{1}-{2}-{3}-{4}".format(
                        args_.split,
                        args_.network,
                        args_.normalize,
                        args_.graph_k,
                        datetime.datetime.now().strftime('%b%d_%H-%M-%S'))

    # Add file handler to logging system to simultaneously log information to console and file
    log_formatter_ = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    file_handler_ = logging.FileHandler("{0}/{1}.log".format(args_.log_path, experiment_str_))
    file_handler_.setFormatter(log_formatter_)
    log.addHandler(file_handler_)

    evaluate(args_)