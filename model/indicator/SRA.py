import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class SimpleResourceAllocation(Indicator):
    def __init__(self,steps:int=2):         
        super().__init__()       
        self.steps = steps 
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
            unit_ra_matrix = np.diag(weight_vector) @ direct_matrix
            init_resource = np.diag(pos_matrix.sum(axis=1))
            self.scores_matrix = init_resource @ np.linalg.matrix_power(unit_ra_matrix,self.steps)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            weight_vector = 1 / (np.asarray(pos_matrix.sum(axis=0) -1).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix = pos_matrix @ ssp.diags(weight_vector) @ pos_matrix.T
            weight_vector = 1 / np.asarray(pos_matrix.sum(axis=1)).squeeze()
            weight_vector[np.isinf(weight_vector)] = 0
            unit_ra_matrix = ssp.diags(weight_vector) @ direct_matrix
            init_resource = ssp.diags(np.asarray(pos_matrix.sum(axis=1)).squeeze())
            self.scores_matrix = init_resource @ unit_ra_matrix**(self.steps)
        # 得到预测分数
        return  self(edge_matrix)