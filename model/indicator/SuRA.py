import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class  SuperposeResourceAllocation(Indicator):
    def __init__(self,alpha=0.2):         
        super().__init__()       
        self.alpha = alpha
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算分数矩阵
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            weight_vector = 1 / (pos_matrix.sum(axis=0) - 1)
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix = pos_matrix @ np.diag(weight_vector) @ pos_matrix.T
            weight_vector = 1 / pos_matrix.sum(axis=1)
            weight_vector[np.isinf(weight_vector)] = 0
            temp_matrix = np.eye(direct_matrix.shape[0]) - self.alpha * np.diag(weight_vector) @ direct_matrix
            self.scores_matrix = direct_matrix @ np.linalg.inv(temp_matrix)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            weight_vector = 1 / (np.asarray(pos_matrix.sum(axis=0) -1).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix= pos_matrix @ ssp.diags(weight_vector) @ pos_matrix.T
            weight_vector = 1 / np.asarray(pos_matrix.sum(axis=1)).squeeze()
            weight_vector[np.isinf(weight_vector)] = 0
            temp_matrix = ssp.eye(direct_matrix.shape[0]) - self.alpha * ssp.diags(weight_vector) @ direct_matrix
            self.scores_matrix = direct_matrix @ ssp.linalg.inv(temp_matrix)
        # 得到预测分数
        return  self(edge_matrix)