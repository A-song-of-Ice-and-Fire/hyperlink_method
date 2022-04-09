import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class NodeEdgeAdamicAdar(Indicator):
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
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            adj_matrix = pos_matrix @ pos_matrix.T
            adj_matrix[np.diag_indices_from(adj_matrix)] = 0
            adj_matrix[adj_matrix != 0] = 1
            weight_vector = 1 / np.log(pos_matrix.sum(axis=1))
            weight_vector[np.isinf(weight_vector)] = 0
            self.scores_matrix = adj_matrix @ np.diag(weight_vector) @ adj_matrix
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            adj_matrix = (pos_matrix @ pos_matrix.T).tolil()
            adj_matrix[np.diag_indices_from(adj_matrix)] = 0
            adj_matrix[adj_matrix != 0] = 1
            weight_vector = 1 / np.log(np.asarray(pos_matrix.sum(axis=1)).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            self.scores_matrix = adj_matrix @ ssp.diags(weight_vector) @ adj_matrix

        # 得到预测分数
        return  self(edge_matrix)