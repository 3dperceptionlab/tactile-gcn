import logging

import torch
from torch_geometric.data import Data
import numpy as np

log = logging.getLogger(__name__)

class ToGraph(object):

    def __init__(self, xTaxels, yTaxels):
        self.m_taxels_x = xTaxels
        self.m_taxels_y = yTaxels

    def __call__(self, sample):

        # Index finger
        log.debug(sample['data_index'])
        index_graph_x_ = torch.tensor(sample['data_index'], dtype=torch.float)
        index_graph_edge_index_ = torch.tensor([[0],[0]], dtype=torch.long)
        index_graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y)), dtype=torch.float)

        data_index_ = Data(x = index_graph_x_,
                            edge_index = index_graph_edge_index_,
                            pos = index_graph_pos_)

        # Middle finger
        middle_graph_x_ = torch.tensor(sample['data_middle'], dtype=torch.float)
        middle_graph_edge_index_ = torch.tensor([[0],[0]], dtype=torch.long)
        middle_graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y)), dtype=torch.float)

        data_middle_ = Data(x = middle_graph_x_,
                            edge_index = middle_graph_edge_index_,
                            pos = middle_graph_pos_)

        # Thumb finger
        thumb_graph_x_ = torch.tensor(sample['data_thumb'], dtype=torch.float)
        thumb_graph_edge_index_ = torch.tensor([[0],[0]], dtype=torch.long)
        thumb_graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y)), dtype=torch.float)

        data_thumb_ = Data(x = thumb_graph_x_,
                            edge_index = thumb_graph_edge_index_,
                            pos = thumb_graph_pos_)


        return {'object': sample['object'],
                'slipped': sample['slipped'],
                'data_index' : data_index_,
                'data_middle': data_middle_,
                'data_thumb' : data_thumb_}

    def __repr__(self):
        return "{}".format(self.__class__.__name__)