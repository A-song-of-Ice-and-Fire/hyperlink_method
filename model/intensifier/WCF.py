import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
)
from .Base import Intensifier

class WeightedCollaborativeFilter(Intensifier):
    def __init__(self,raw_indicator_str:str="NNAA"):         
        super().__init__(raw_indicator_str)        
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
            adj_matrix[adj_matrix!=0] = 1

        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            adj_matrix = (pos_matrix @ pos_matrix.T).tolil()
            adj_matrix[adj_matrix !=0 ]=1
            adj_matrix = adj_matrix.tocsr()
        self.scores_matrix = (adj_matrix * self.raw_matrix) @ self.raw_matrix + ((adj_matrix * self.raw_matrix) @ self.raw_matrix).T

        # 得到预测分数
        return  self(edge_matrix)