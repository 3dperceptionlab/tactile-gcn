import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


def plot_edges(axis, pos, edgeIndex):

    for i in range(len(edgeIndex[0])):

        origin_tuple_ = (pos[0][edgeIndex[0][i]],
                        pos[0][edgeIndex[1][i]])
        end_tuple_ = (pos[1][edgeIndex[0][i]],
                        pos[1][edgeIndex[1][i]])

        log.info(origin_tuple_)
        log.info(end_tuple_)

        axis.plot(origin_tuple_, end_tuple_, '-k')

def plot_graph(sample):

    levels_ = np.linspace(0, 4096, 64)
    xtics_ = np.arange(-4.0, 5.0, step=1.0)
    ytics_ = np.arange(-6.0, 6.0, step=1.0)

    figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
    (ax0_, ax1_, ax2_) = axes_

    cf0_ = ax0_.scatter(sample['data_index'].pos[0], sample['data_index'].pos[1], s=sample['data_index'].x / 3.0, c=sample['data_index'].x, vmin=0, vmax=4096)
    plot_edges(ax0_, sample['data_index'].pos, sample['data_index'].edge_index)
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger')
    ax0_.set_ylabel('Vertical Position [mm]')
    ax0_.set_xlabel('Horizontal Position [mm]')

    cf1_ = ax1_.scatter(sample['data_middle'].pos[0], sample['data_middle'].pos[1], s=sample['data_middle'].x / 3.0, c=sample['data_middle'].x, vmin=0, vmax=4096)
    plot_edges(ax1_, sample['data_middle'].pos, sample['data_middle'].edge_index)
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger')
    ax1_.set_ylabel('Vertical Position [mm]')
    ax1_.set_xlabel('Horizontal Position [mm]')

    cf2_ = ax2_.scatter(sample['data_thumb'].pos[0], sample['data_thumb'].pos[1], s=sample['data_thumb'].x / 3.0, c=sample['data_thumb'].x, vmin=0, vmax=4096)
    plot_edges(ax2_, sample['data_thumb'].pos, sample['data_thumb'].edge_index)
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger')
    ax2_.set_ylabel('Vertical Position [mm]')
    ax2_.set_xlabel('Horizontal Position [mm]')

    figure_.tight_layout()

    figure_.colorbar(cf2_, ax=axes_.ravel().tolist(), orientation='horizontal')

    plt.show()