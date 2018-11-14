import logging

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset

import transforms.tograph

log = logging.getLogger(__name__)

class BioTacSp(InMemoryDataset):

  def __init__(self, root, transform=None, pre_transform=None):
    super(BioTacSp, self).__init__(root, transform, pre_transform)

    self.data, self.slices = torch.load(self.processed_paths[0])

    # Compute class weights for sampling
    self.class_weights = np.zeros(2)
    for i in range(len(self.data['y'])):
      self.class_weights[self.data['y'][i]] += 1
    self.class_weights /= len(self.data['y'])

  @property
  def raw_file_names(self):
    return ['biotac-palmdown-grasps.csv', 'biotac-palmside-grasps.csv']

  @property
  def processed_file_names(self):
    return ['biotacsp.pt']

  def download(self):

    url_ = "https://github.com/yayaneath/biotac-sp-images"

    raise RuntimeError(
      "Dataset not found. Please download {} from {} and move it to {}".format(
        str(self.raw_file_names),
        url_,
        self.raw_dir))

  def process(self):
    
    transform_tograph_ = transforms.tograph.ToGraph()

    data_list_ = []

    for f in range(len(self.raw_paths)):

      log.info("Reading CSV file {0}".format(self.raw_paths[f]))

      grasps_ = pd.read_csv(self.raw_paths[f])

      for i in range(len(grasps_)):
      
        sample_ = self._sample_from_csv(grasps_, i)
        sample_ = transform_tograph_(sample_)

        if self.pre_transform is not None:
          sample_ = self.pre_transform(sample_)

        data_list_.append(sample_)

    data_ = self.collate(data_list_)

    torch.save(data_, self.processed_paths[0])

  def _sample_from_csv(self, grasps, idx):

    sample_ = grasps.iloc[idx]
    
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

    return sample_