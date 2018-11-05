import logging

import torch
import torch.utils.data

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

class BioTacSpDataset(torch.utils.data.Dataset):
	"""BioTacSp Grasping Dataset"""

	def __init__(self, csvFile, transform=None):
		"""
		Args:
			csvFile (string): Path to the CSV file with annotations.
		"""
		self.m_csv_file = csvFile
		self.m_grasps = pd.read_csv("data/" + self.m_csv_file)
		self.m_transform = transform

	def __len__(self):
		
		return len(self.m_grasps)

	def __getitem__(self, idx):

		sample_ = self.m_grasps.iloc[idx]

		object_ = sample_.iloc[0]
		slipped_ = sample_.iloc[1]
		data_index_ = np.copy(sample_.iloc[2:26]).astype(np.int, copy=False)
		data_middle_ = np.copy(sample_.iloc[26:50]).astype(np.int, copy=False)
		data_thumb_ = np.copy(sample_.iloc[50:75]).astype(np.int, copy=False)

		sample_ = {'object': object_,
							'slipped': slipped_,
							'data_index': data_index_,
							'data_middle': data_middle_,
							'data_thumb': data_thumb_}

		if self.m_transform:
			sample_ = self.m_transform(sample_)

		return sample_

	def __repr__(self):

		return "Dataset loader for BiotacSp with {0} entries in {1}".format(
			self.__len__(),
			self.m_csv_file
		)