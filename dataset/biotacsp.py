import logging

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset

import transforms.tograph

log = logging.getLogger(__name__)

class BioTacSp(InMemoryDataset):

  def __init__(self, root, k, split="train", normalize=True, csvs=None, transform=None, pre_transform=None):

    self.split = split
    self.csvs = csvs
    self.k = k
    self.normalize = normalize
    self.mins = []
    self.maxs = []

    super(BioTacSp, self).__init__(root, transform, pre_transform)

    self.data, self.slices = torch.load(self.processed_paths[0])

    # Compute class weights for sampling
    self.class_weights = np.zeros(2)
    for i in range(len(self.data['y'])):
      self.class_weights[self.data['y'][i]] += 1
    self.class_weights /= len(self.data['y'])

  @property
  def raw_file_names(self):
    if (self.split == "train"):
      return ['biotac-palmdown-grasps.csv', 'biotac-palmside-grasps.csv', 'palm_45.csv']
    elif (self.split == "test"):
      return ['palm_45_test.csv', 'palm_down_test.csv', 'palm_side_test.csv']
    elif (self.split == None):
      return self.csvs

  @property
  def processed_file_names(self):
    if (self.split == "train"):
      return ["biotacsp_{0}.pt".format(self.k)]
    elif (self.split == "test"):
      return ["biotacsp_test_{0}.pt".format(self.k)]
    elif (self.split == None):
      return ['biotacsp_' + ''.join(self.csvs) + '.pt']

  def download(self):

    url_ = "https://github.com/yayaneath/biotac-sp-images"

    raise RuntimeError(
      "Dataset not found. Please download {} from {} and move it to {}".format(
        str(self.raw_file_names),
        url_,
        self.raw_dir))

  def process(self):

    transform_tograph_ = transforms.tograph.ToGraph(self.k)

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

    if self.normalize: # Feature scaling
      raw_dataset_np_ = np.array([sample.x.numpy() for sample in data_list_])

      self.mins = np.min(raw_dataset_np_, axis=(0, 1))
      self.maxs = np.max(raw_dataset_np_, axis=(0, 1))

      for i in range(len(data_list_)):
        data_list_[i].x = torch.from_numpy((data_list_[i].x.numpy() - self.mins) / (self.maxs - self.mins))

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
