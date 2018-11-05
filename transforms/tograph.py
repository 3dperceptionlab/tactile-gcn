import logging

import torch
from torch_geometric.data import Data
import numpy as np

log = logging.getLogger(__name__)

class ToGraph(object):

    def __init__(self):

        self.m_taxels_x = [-3.0, -2.0, -4.0, -2.5, -1.5, -4.0, -2.5, -0.5, -2.0, -2.5, 3.0, 2.0, 4.0, 2.5, 1.5, 4.0, 2.5, 0.5, 2.0, 2.5, 0.0, -1.0, 1.0, 0.0]
        self.m_taxels_y = [5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 5.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 3.0, 2.0, 2.0, 0.0]
        self.m_edge_origins =   [0, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 10, 11, 11, 12, 13, 13, 13, 14, 14, 14, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23]
        self.m_edge_ends =      [1, 0, 20, 3, 2, 4, 21, 23, 3, 6, 7, 23, 6, 5, 4, 7, 8, 4, 6, 8, 17, 6, 7, 9, 8, 11, 10, 20, 13, 12, 23, 14, 13, 16, 17, 16, 15, 14, 17, 18, 14, 16, 18, 7, 17, 16, 19, 18, 1, 11, 21, 22, 3, 20, 22, 23, 13, 20, 21, 23, 21, 22, 3, 13, 4, 14]

    def __call__(self, sample):

        # Index finger
        index_graph_x_ = torch.tensor(sample['data_index'], dtype=torch.float)
        index_graph_edge_index_ = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
        index_graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y)), dtype=torch.float)

        data_index_ = Data(x = index_graph_x_,
                            edge_index = index_graph_edge_index_,
                            pos = index_graph_pos_)

        # Middle finger
        middle_graph_x_ = torch.tensor(sample['data_middle'], dtype=torch.float)
        middle_graph_edge_index_ = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
        middle_graph_pos_ = torch.tensor(np.vstack((self.m_taxels_x, self.m_taxels_y)), dtype=torch.float)

        data_middle_ = Data(x = middle_graph_x_,
                            edge_index = middle_graph_edge_index_,
                            pos = middle_graph_pos_)

        # Thumb finger
        thumb_graph_x_ = torch.tensor(sample['data_thumb'], dtype=torch.float)
        thumb_graph_edge_index_ = torch.tensor([self.m_edge_origins, self.m_edge_ends], dtype=torch.long)
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