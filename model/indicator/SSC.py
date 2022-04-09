import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class SuperposeSpreadComb(Indicator):
    def __init__(self,steps:int=2,alpha:float=0.2):         
        super().__init__()       
        self.steps = steps 
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
            weight_vector = 1 / (pos_matrix.sum(axis=0)-1)
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix = pos_matrix @ np.diag(weight_vector) @ pos_matrix.T
            weight_vector = 1 / pos_matrix.sum(axis=1)
            weight_vector[np.isinf(weight_vector)] = 0
            unit_sc_matrix = (np.diag(weight_vector) ** self.alpha) @ direct_matrix
            if self.alpha<1:
                unit_sc_matrix = unit_sc_matrix @ (np.diag(weight_vector) ** (1 - self.alpha))
            init_resource_matrix = np.diag(pos_matrix.sum(axis=1))
            self.scores_matrix = 0
            temp_matrix = init_resource_matrix
            for _ in range(self.steps):
                temp_matrix = temp_matrix @ unit_sc_matrix
                self.scores_matrix += temp_matrix
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            weight_vector = 1 / (np.asarray(pos_matrix.sum(axis=0) -1).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix  = pos_matrix @ ssp.diags(weight_vector) @ pos_matrix.T
            weight_vector = 1 / np.asarray(pos_matrix.sum(axis=1)).squeeze()
            weight_vector[np.isinf(weight_vector)] = 0
            unit_sc_matrix = (ssp.diags(weight_vector).power(self.alpha)) @ direct_matrix
            if self.alpha < 1:
                unit_sc_matrix = unit_sc_matrix @ ssp.diags(weight_vector).power(1 - self.alpha)
            init_resource_matrix = ssp.diags(np.asarray(pos_matrix.sum(axis=1)).squeeze())
            self.scores_matrix = 0
            temp_matrix = init_resource_matrix
            for _ in range(self.steps):
                temp_matrix = temp_matrix @ unit_sc_matrix
                self.scores_matrix += temp_matrix
        # 得到预测分数
        return  self(edge_matrix)