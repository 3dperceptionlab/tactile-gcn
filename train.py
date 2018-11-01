import logging
import sys

import torch

from torch_geometric.data import Data

import loader.biotacsp_loader
import transforms.tograph
import utils.plotgraph

log = logging.getLogger(__name__)

def train():

    CSV_FILE = "biotac-palmdown-grasps.csv"

    biotacsp_dataset_ = loader.biotacsp_loader.BioTacSpDataset(csvFile=CSV_FILE)

    # Transformations
    transform_tograph_ = transforms.tograph.ToGraph(biotacsp_dataset_.m_taxels_x, biotacsp_dataset_.m_taxels_y)

    log.info(biotacsp_dataset_)

    for i in range(len(biotacsp_dataset_)):

        sample_ = biotacsp_dataset_[i]
        log.info(sample_)
        biotacsp_dataset_.plot(sample_)

        graph_sample_ = transform_tograph_(sample_)

        utils.plotgraph.plot_graph(graph_sample_)




if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    train()