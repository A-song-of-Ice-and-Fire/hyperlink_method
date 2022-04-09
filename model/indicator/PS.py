import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class ProbabilisticSpread(Indicator):
    def __init__(self):         
        super().__init__()       
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
            unit_ps_matrix = direct_matrix @ np.diag(weight_vector)

            # 初始资源可以有很多种
            # 节点单位资源
            init_resource_matrix = np.eye(pos_matrix.shape[0])
            one_step_matrix = init_resource_matrix @ unit_ps_matrix

            self.scores_matrix = one_step_matrix + ( one_step_matrix @ unit_ps_matrix ) 
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            weight_vector = 1 / (np.asarray(pos_matrix.sum(axis=0) -1).squeeze())
            weight_vector[np.isinf(weight_vector)] = 0
            direct_matrix  = pos_matrix @ ssp.diags(weight_vector) @ pos_matrix.T
            weight_vector = 1 / np.asarray(pos_matrix.sum(axis=1)).squeeze()
            weight_vector[np.isinf(weight_vector)] = 0
            unit_ps_matrix = direct_matrix @ ssp.diags(weight_vector)
            
            # 节点单位资源
            init_resource_matrix = ssp.eye(pos_matrix.shape[0])
            one_step_matrix = init_resource_matrix @ unit_ps_matrix

            self.scores_matrix = one_step_matrix + ( one_step_matrix @ unit_ps_matrix )
        # 得到预测分数
        return  self(edge_matrix)