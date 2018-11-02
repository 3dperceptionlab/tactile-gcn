import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)

def plot_contour(sample, xTaxels, yTaxels):

    x_i_ = np.linspace(min(xTaxels), max(xTaxels))
    y_i_ = np.linspace(min(yTaxels), max(yTaxels))
    x_, y_ = np.meshgrid(x_i_, y_i_)
    levels_ = np.linspace(0, 4096, 64)
    xtics_ = np.arange(-4.0, 5.0, step=1.0)
    ytics_ = np.arange(-6.0, 6.0, step=1.0)

    figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
    (ax0_, ax1_, ax2_) = axes_

    z_ = matplotlib.mlab.griddata(xTaxels, yTaxels, sample['data_index'], x_i_, y_i_, interp='linear')

    cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
    ax0_.set(aspect='equal')
    ax0_.grid()
    ax0_.set_xticks(xtics_)
    ax0_.set_yticks(ytics_)
    ax0_.set_title('Index Finger')
    ax0_.set_ylabel('Vertical Position [mm]')
    ax0_.set_xlabel('Horizontal Position [mm]')

    z_ = matplotlib.mlab.griddata(xTaxels, yTaxels, sample['data_middle'], x_i_, y_i_, interp='linear')

    cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
    ax1_.set(aspect='equal')
    ax1_.grid()
    ax1_.set_xticks(xtics_)
    ax1_.set_yticks(ytics_)
    ax1_.set_title('Middle Finger')
    ax1_.set_xlabel('Horizontal Position [mm]')

    z_ = matplotlib.mlab.griddata(xTaxels, yTaxels, sample['data_thumb'], x_i_, y_i_, interp='linear')

    cf2_ = ax2_.contourf(x_, y_, z_, 20, levels=levels_)
    ax2_.set(aspect='equal')
    ax2_.grid()
    ax2_.set_xticks(xtics_)
    ax2_.set_yticks(ytics_)
    ax2_.set_title('Thumb Finger')
    ax2_.set_xlabel('Horizontal Position [mm]')

    figure_.tight_layout()

    figure_.colorbar(cf2_, ax=axes_.ravel().tolist(), orientation='horizontal')

    plt.show()