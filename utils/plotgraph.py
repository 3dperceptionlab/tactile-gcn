import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.mlab
import mpl_toolkits.mplot3d
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

def plot_edges_3d(axis, pos, edgeIndex):

    for i in range(len(edgeIndex[0])):


        x_tuple_ = (pos[edgeIndex[0][i]][0],
                        pos[edgeIndex[1][i]][0])
        y_tuple_ = (pos[edgeIndex[0][i]][1],
                        pos[edgeIndex[1][i]][1])
        z_tuple_ = (pos[edgeIndex[0][i]][2],
                        pos[edgeIndex[1][i]][2])


        axis.plot(x_tuple_, y_tuple_, z_tuple_, '-k')


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
    ax0_.set_ylabel('Vertical Position [inch]')
    ax0_.set_xlabel('Horizontal Position [inch]')

    z_ = matplotlib.mlab.griddata(sample['data_middle'].pos[0], sample['data_middle'].pos[1], sample['data_middle'].x, x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.scatter(sample['data_middle'].pos[0], sample['data_middle'].pos[1], s=sample['data_middle'].x / 3.0, c=sample['data_middle'].x, vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax1_, sample['data_middle'].pos, sample['data_middle'].edge_index)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger')
    ax1_.set_xlabel('Horizontal Position [inch]')

    z_ = matplotlib.mlab.griddata(sample['data_thumb'].pos[0], sample['data_thumb'].pos[1], sample['data_thumb'].x, x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.scatter(sample['data_thumb'].pos[0], sample['data_thumb'].pos[1], s=sample['data_thumb'].x / 3.0, c=sample['data_thumb'].x, vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax2_, sample['data_thumb'].pos, sample['data_thumb'].edge_index)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger')
    ax2_.set_xlabel('Horizontal Position [inch]')

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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
    (ax0_, ax1_, ax2_) = axes_

    ax0_.tick_params(axis='both', which='major', labelsize=12)
    ax1_.tick_params(axis='both', which='major', labelsize=12)
    ax2_.tick_params(axis='both', which='major', labelsize=12)

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 0], x_i_, y_i_, interp='linear')

    cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
    ax0_.scatter(pos[:, 0], pos[:, 1], s=x[:, 0] / 3.0, c=x[:, 0], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax0_, pos, edgeIndex)
    ax0_.set(aspect='equal')
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger', fontsize=32)
    ax0_.set_ylabel('Vertical Position [inch]', fontsize=32)
    ax0_.set_xlabel('Horizontal Position [inch]', fontsize=32)

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 1], x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.scatter(pos[:, 0], pos[:, 1], s=x[:, 1] / 3.0, c=x[:, 1], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax1_, pos, edgeIndex)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger', fontsize=32)
    ax1_.set_xlabel('Horizontal Position [inch]', fontsize=32)

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 2], x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.scatter(pos[:, 0], pos[:, 1], s=x[:, 2] / 3.0, c=x[:, 2], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax2_, pos, edgeIndex)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger', fontsize=32)
    ax2_.set_xlabel('Horizontal Position [inch]', fontsize=32)

    figure_.tight_layout()

    cbar_ = figure_.colorbar(cf2_, ax=axes_.ravel().tolist(), orientation='horizontal')
    cbar_.ax.tick_params(labelsize=28)

    plt.show()

def plot_contourgraph_batch_paper(pos, x, y, edgeIndex):

    x_i_ = np.linspace(min(pos[:, 0]), max(pos[:, 0]))
    y_i_ = np.linspace(min(pos[:, 1]), max(pos[:, 1]))
    x_, y_ = np.meshgrid(x_i_, y_i_)
    levels_ = np.linspace(0, 4096, 64)
    xtics_ = np.arange(-0.3, 0.4, step=0.1)
    ytics_ = np.arange(-0.4, 0.5, step=0.1)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    figure_, axes_ = plt.subplots(ncols=1, sharex=True, sharey=True, figsize=(15, 15))
    (ax0_) = axes_

    ax0_.tick_params(axis='both', which='major', labelsize=24)

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 0], x_i_, y_i_, interp='linear')

    cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
    ax0_.scatter(pos[:, 0], pos[:, 1], s=x[:, 0] / 3.0, c=x[:, 0], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax0_, pos, edgeIndex)
    ax0_.set(aspect='equal')
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger', fontsize=32)
    ax0_.set_ylabel('Vertical Position [inch]', fontsize=32)
    ax0_.set_xlabel('Horizontal Position [inch]', fontsize=32)
    cbar_ = figure_.colorbar(cf0_, ax=ax0_, orientation='vertical')
    cbar_.ax.tick_params(labelsize=22)

    figure_.savefig('contour_graph_index_paper.png',dpi=300)

    plt.show()

    figure_, axes_ = plt.subplots(ncols=1, sharex=True, sharey=True)
    (ax1_) = axes_

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 1], x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.scatter(pos[:, 0], pos[:, 1], s=x[:, 1] / 3.0, c=x[:, 1], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax1_, pos, edgeIndex)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger', fontsize=20)
    ax1_.set_ylabel('Vertical Position [inch]', fontsize=32)
    ax1_.set_xlabel('Horizontal Position [inch]', fontsize=32)
    figure_.colorbar(cf1_, ax=ax1_, orientation='vertical')

    figure_.savefig('contour_graph_middle_paper.png',dpi=300)

    plt.show()

    figure_, axes_ = plt.subplots(ncols=1, sharex=True, sharey=True)
    (ax2_) = axes_

    z_ = matplotlib.mlab.griddata(pos[:, 0], pos[:, 1], x[:, 2], x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.scatter(pos[:, 0], pos[:, 1], s=x[:, 2] / 3.0, c=x[:, 2], vmin=0, vmax=4096, alpha=0.5, edgecolors='b', linewidth=1)
    plot_edges(ax2_, pos, edgeIndex)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger', fontsize=20)
    ax2_.set_ylabel('Vertical Position [inch]', fontsize=20)
    ax2_.set_xlabel('Horizontal Position [inch]', fontsize=20)
    figure_.colorbar(cf2_, ax=ax2_, orientation='vertical')

    figure_.savefig('contour_graph_thumb_paper.png',dpi=300)

    plt.show()

def plot_graph_3d(pos, x, y, edgeIndex):

    figure_ = plt.figure(figsize=(16, 16))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ax_ = mpl_toolkits.mplot3d.Axes3D(figure_)

    ax_.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=x[:, 1] / 6.0)

    plot_edges_3d(ax_, pos, edgeIndex)

    for t in ax_.xaxis.get_major_ticks(): t.label.set_fontsize(1)
    for t in ax_.yaxis.get_major_ticks(): t.label.set_fontsize(1)
    for t in ax_.zaxis.get_major_ticks(): t.label.set_fontsize(1)

    ax_.set_xlabel('X', fontsize=48)
    ax_.set_xlim(-0.3, 0.3)
    ax_.set_ylabel('Y', fontsize=48)
    ax_.set_ylim(-0.3, 0.3)
    ax_.set_zlabel('Z', fontsize=48)
    ax_.set_zlim(-0.3, 0.0)

    for i in range(len(pos[:, 0])):
        x_ = pos[i][0].item()
        y_ = pos[i][1].item()
        z_ = pos[i][2].item()
        ax_.scatter(x_, y_, z_, 'bo')
        t_ = ax_.text(x_ * (1 + 0.1), y_ * (1 + 0.1), z_ * (1 + 0.1), i + 1, color='white',  fontsize=32, zorder=100)
        t_.set_bbox(dict(facecolor='green', alpha=0.5, edgecolor='green'))

    figure_.savefig('graph_3d.png',dpi=300)

    plt.show()
