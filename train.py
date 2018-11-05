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
    biotacsp_loader_ = DataLoader(biotacsp_dataset_, batch_size=2, shuffle=False, num_workers=1)

    # Transformations

    log.info(biotacsp_dataset_)

    for batch in biotacsp_loader_:
        
        log.info(batch)
        log.info("Batch size {}".format(batch.num_graphs))

        log.info(batch['edge_index'][0])
        log.info(batch['edge_index'][1])
        log.info(batch['y'])

        for i in range(batch.num_graphs):
                pos_ = batch['pos'][i*2: 2 + i*2, :]
                x_ = batch['x'][i*3: 3 + i*3, :]
                y_ = batch['y'][i:i]
                edge_index_ = batch['edge_index'][:,i*66:66+i*66] - 3*i

                utils.plotgraph.plot_contourgraph_batch(pos_, x_, y_, edge_index_)

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    train()