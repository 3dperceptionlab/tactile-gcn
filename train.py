import logging
import sys

import torch
import torch.utils.data.dataloader

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import loader.biotacsp_loader
import dataset.biotacsp
import transforms.tograph
import utils.plotcontour
import utils.plotgraph

log = logging.getLogger(__name__)

def train():

    CSV_FILE = "biotac-palmdown-grasps.csv"

    transform_tograph_ = transforms.tograph.ToGraph()

    biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='~/Workspace/3dpl/tactile-gcn/data/biotacsp')
    biotacsp_loader_ = DataLoader(biotacsp_dataset_, batch_size=4, shuffle=True, num_workers=4)

    # Transformations

    log.info(biotacsp_dataset_)

    for batch in biotacsp_loader_:
        
        batch
        #sample_ = biotacsp_dataset_[i]
        #log.info(sample_)
        #utils.plotcontour.plot_contour(sample_, biotacsp_dataset_.m_taxels_x, biotacsp_dataset_.m_taxels_y)

        #graph_sample_ = transform_tograph_(sample_)

        #utils.plotgraph.plot_contourgraph(graph_sample_)
        #utils.plotgraph.plot_graph(graph_sample_)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    train()