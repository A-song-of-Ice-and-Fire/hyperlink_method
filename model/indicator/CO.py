import numpy as np
import scipy.sparse as ssp

from . import (
    CommonNeighbor,
    CoexistEdge
)
from .Base import Indicator
from typing import (
    Union,
    Optional
)

class CommonObject(Indicator):
    def __init__(self):         
        super().__init__()        
        self.scores_matrix = None        
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算邻接矩阵
        common_neighbor = CommonNeighbor()
        coexist_edge = CoexistEdge()
        common_neighbor.train(edge_matrix,obvious_edge_index)
        coexist_edge.train(edge_matrix,obvious_edge_index)
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            adj_matrix = pos_matrix @ pos_matrix.T
            adj_matrix[np.diag_indices_from(adj_matrix)] = 0
            adj_matrix[adj_matrix != 0] = 1
            degree_by_node = adj_matrix.sum(axis=1).mean()
            degree_by_edge = pos_matrix.sum(axis=1).mean()
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            adj_matrix = (pos_matrix @ pos_matrix.T).tolil()
            adj_matrix[np.diag_indices_from(adj_matrix)] = 0
            adj_matrix[adj_matrix != 0] = 1
            degree_by_node = np.asarray(adj_matrix.sum(axis=1)).squeeze().mean()
            degree_by_edge = np.asarray(pos_matrix.sum(axis=1)).squeeze().mean()

        ratio = degree_by_edge / degree_by_node

        alpha = 0
        #self.scores_matrix = (ratio / (1+ratio))*common_neighbor.scores_matrix + (1 / ( 1 + ratio )) * coexist_edge.scores_matrix
        self.scores_matrix = alpha * common_neighbor.scores_matrix + (1-alpha)*coexist_edge.scores_matrix
        # 得到预测分数
        return  self(edge_matrix)