import logging

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.mlab
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

class BioTacSpDataset(torch.utils.data.Dataset):
	"""BioTacSp Grasping Dataset"""

	def __init__(self, csvFile):
		"""
		Args:
			csvFile (string): Path to the CSV file with annotations.
		"""
		self.m_csv_file = csvFile
		self.m_grasps = pd.read_csv("data/" + self.m_csv_file)

		self.m_taxels_x = [-3.0, -2.0, -4.0, -2.5, -1.5, -4.0, -2.5, -0.5, -2.0, -2.5, 3.0, 2.0, 4.0, 2.5, 1.5, 4.0, 2.5, 0.5, 2.0, 2.5, 0.0, -1.0, 1.0, 0.0]
		self.m_taxels_y = [5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 3.0, 2.0, 2.0, 0.0]

	def __len__(self):
		
		return len(self.m_grasps)

	def __getitem__(self, idx):

		sample_ = self.m_grasps.iloc[idx]

		object_ = sample_.iloc[0]
		slipped_ = sample_.iloc[1]
		data_index_ = np.copy(sample_.iloc[2:26]).astype(np.int, copy=False)
		data_middle_ = np.copy(sample_.iloc[26:50]).astype(np.int, copy=False)
		data_thumb_ = np.copy(sample_.iloc[50:75]).astype(np.int, copy=False)

		return {'object': object_,
				'slipped': slipped_,
				'data_index': data_index_,
				'data_middle': data_middle_,
				'data_thumb': data_thumb_}

	def __repr__(self):

		return "Dataset loader for BiotacSp with {0} entries in {1}".format(
			self.__len__(),
			self.m_csv_file
		)

	def plot(self, sample):

		x_i_ = np.linspace(min(self.m_taxels_x), max(self.m_taxels_x))
		y_i_ = np.linspace(min(self.m_taxels_y), max(self.m_taxels_y))
		x_, y_ = np.meshgrid(x_i_, y_i_)
		levels_ = np.linspace(0, 4096, 64)
		xtics_ = np.arange(-4.0, 5.0, step=1.0)
		ytics_ = np.arange(-6.0, 6.0, step=1.0)

		figure_, axes_ = plt.subplots(ncols=3, sharex=True, sharey=True)
		(ax0_, ax1_, ax2_) = axes_

		z_ = matplotlib.mlab.griddata(self.m_taxels_x, self.m_taxels_y, sample['data_index'], x_i_, y_i_, interp='linear')

		cf0_ = ax0_.contourf(x_, y_, z_, 20, levels=levels_)
		ax0_.set(aspect='equal')
		ax0_.grid()
		ax0_.set_xticks(xtics_)
		ax0_.set_yticks(ytics_)
		ax0_.set_title('Index Finger')
		ax0_.set_ylabel('Vertical Position [mm]')
		ax0_.set_xlabel('Horizontal Position [mm]')

		z_ = matplotlib.mlab.griddata(self.m_taxels_x, self.m_taxels_y, sample['data_middle'], x_i_, y_i_, interp='linear')

		cf1_ = ax1_.contourf(x_, y_, z_, 20, levels=levels_)
		ax1_.set(aspect='equal')
		ax1_.grid()
		ax1_.set_xticks(xtics_)
		ax1_.set_yticks(ytics_)
		ax1_.set_title('Middle Finger')
		ax1_.set_xlabel('Horizontal Position [mm]')

		z_ = matplotlib.mlab.griddata(self.m_taxels_x, self.m_taxels_y, sample['data_thumb'], x_i_, y_i_, interp='linear')

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
