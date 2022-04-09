import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
)

class EdgeEdgeAdamicAdar(Indicator):
    def __init__(self):         
        super().__init__()               
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算邻接矩阵
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            edge_adj_matrix = pos_matrix.T @ pos_matrix
            edge_adj_matrix[np.diag_indices_from(edge_adj_matrix)] = 0
            edge_adj_matrix[edge_adj_matrix != 0] = 1
            weight_vector = 1 / np.log(edge_adj_matrix.sum(axis=0))
            weight_vector[np.isinf(weight_vector)] = 0
            self.scores_matrix = pos_matrix @ np.diag(weight_vector) @ pos_matrix.T
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_adj_matrix = (pos_matrix.T @ pos_matrix).tolil()
            edge_adj_matrix[np.diag_indices_from(edge_adj_matrix)] = 0
            edge_adj_matrix[edge_adj_matrix != 0] = 1
            weight_vector = 1 / np.log(np.asarray(edge_adj_matrix.sum(axis=0)).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            self.scores_matrix = pos_matrix @ ssp.diags(weight_vector) @ pos_matrix.T

        # 得到预测分数
        return  self(edge_matrix)
