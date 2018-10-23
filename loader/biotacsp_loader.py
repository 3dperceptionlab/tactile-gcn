import torch
import torch.utils.data

import pandas as pd

class BioTacSpDataset(torch.utils.data.Dataset):
	"""BioTacSp Grasping Dataset"""

	def __init__(self, csvFile):
		"""
		Args:
			csvFile (string): Path to the CSV file with annotations.
		"""
		self.m_csv_file = csvFile
		self.m_grasps = pd.read_csv("data/" + self.m_csv_file)

	def __len__(self):
		
		return len(self.m_grasps)

	def __getitem__(self, idx):

		return self.m_grasps.iloc[idx]

	def __repr__(self):

		return "Dataset loader for BiotacSp with {0} entries in {1}".format(
			self.__len__(),
			self.m_csv_file
		)