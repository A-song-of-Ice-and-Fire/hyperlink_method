import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional
)

class Density(Indicator):
    def __init__(self):         
        super().__init__()        
        self.scores_matrix = None        
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算共存矩阵
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            coexist_matrix = pos_matrix @ pos_matrix.T
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            coexist_matrix[coexist_matrix != 0] = 1
            self.scores_matrix = coexist_matrix
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            coexist_matrix = (pos_matrix @ pos_matrix.T).tolil()
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            coexist_matrix[coexist_matrix != 0] = 1
            self.scores_matrix = coexist_matrix

        # 得到预测分数
        return  self(edge_matrix)