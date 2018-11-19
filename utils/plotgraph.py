import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


def plot_edges(axis, pos, edgeIndex):

    for i in range(len(edgeIndex[0])):

        
        origin_tuple_ = (pos[edgeIndex[0][i]][0],
                        pos[edgeIndex[1][i]][0])
        end_tuple_ = (pos[edgeIndex[0][i]][1],
                        pos[edgeIndex[1][i]][1])

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

def plot_contourgraph(sample):

    x_i_ = np.linspace(min(sample['data_index'].pos[0]), max(sample['data_index'].pos[0]))
    y_i_ = np.linspace(min(sample['data_index'].pos[1]), max(sample['data_index'].pos[1]))
    x_, y_ = np.meshgrid(x_i_, y_i_)
    levels_ = np.linspace(0, 4096, 64)
    xtics_ = np.arange(-0.3, 0.4, step=0.1)
    ytics_ = np.arange(-0.4, 0.5, step=0.1)

    figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
    (ax0_, ax1_, ax2_) = axes_

    z_ = matplotlib.mlab.griddata(sample['data_index'].pos[0], sample['data_index'].pos[1], sample['data_index'].x, x_i_, y_i_, interp='linear')

    cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
    ax0_.scatter(sample['data_index'].pos[0], sample['data_index'].pos[1], s=sample['data_index'].x / 3.0, c=sample['data_index'].x, vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax0_, sample['data_index'].pos, sample['data_index'].edge_index)
    ax0_.set(aspect='equal')
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger')
    ax0_.set_ylabel('Vertical Position [cm]')
    ax0_.set_xlabel('Horizontal Position [cm]')

    z_ = matplotlib.mlab.griddata(sample['data_middle'].pos[0], sample['data_middle'].pos[1], sample['data_middle'].x, x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.scatter(sample['data_middle'].pos[0], sample['data_middle'].pos[1], s=sample['data_middle'].x / 3.0, c=sample['data_middle'].x, vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax1_, sample['data_middle'].pos, sample['data_middle'].edge_index)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger')
    ax1_.set_xlabel('Horizontal Position [cm]')

    z_ = matplotlib.mlab.griddata(sample['data_thumb'].pos[0], sample['data_thumb'].pos[1], sample['data_thumb'].x, x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.scatter(sample['data_thumb'].pos[0], sample['data_thumb'].pos[1], s=sample['data_thumb'].x / 3.0, c=sample['data_thumb'].x, vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax2_, sample['data_thumb'].pos, sample['data_thumb'].edge_index)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger')
    ax2_.set_xlabel('Horizontal Position [cm]')

    figure_.tight_layout()

    figure_.colorbar(cf2_, ax=axes_.ravel().tolist(), orientation='horizontal')

    plt.show()

def plot_contourgraph_batch(pos, x, y, edgeIndex):

    x_i_ = np.linspace(min(pos[:, 0]), max(pos[:, 0]))
    y_i_ = np.linspace(min(pos[:, 1]), max(pos[:, 1]))
    x_, y_ = np.meshgrid(x_i_, y_i_)
    levels_ = np.linspace(0, 4096, 64)
    xtics_ = np.arange(-0.3, 0.4, step=0.1)
    ytics_ = np.arange(-0.4, 0.5, step=0.1)

    figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
    (ax0_, ax1_, ax2_) = axes_
    
    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 0], x_i_, y_i_, interp='linear')

    cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
    ax0_.scatter(pos[:, 0], pos[:, 1], s=x[:, 0] / 3.0, c=x[:, 0], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax0_, pos, edgeIndex)
    ax0_.set(aspect='equal')
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger')
    ax0_.set_ylabel('Vertical Position [cm]')
    ax0_.set_xlabel('Horizontal Position [cm]')

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 1], x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.scatter(pos[:, 0], pos[:, 1], s=x[:, 1] / 3.0, c=x[:, 1], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax1_, pos, edgeIndex)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger')
    ax1_.set_xlabel('Horizontal Position [cm]')

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 2], x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.scatter(pos[:, 0], pos[:, 1], s=x[:, 2] / 3.0, c=x[:, 2], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax2_, pos, edgeIndex)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger')
    ax2_.set_xlabel('Horizontal Position [cm]')

    figure_.tight_layout()

    figure_.colorbar(cf2_, ax=axes_.ravel().tolist(), orientation='horizontal')

    plt.show()