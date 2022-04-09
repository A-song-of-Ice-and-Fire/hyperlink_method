import numpy as np
import scipy.sparse as ssp
from itertools import permutations
from model.indicator.Base import Indicator
from .LRW import LocalRandomWalk
from ..module import Matrix
from typing import (
    Union,
    Optional,
    List
)

class SuperposedLocalRandomWalk(LocalRandomWalk): # 此方法时间复杂度可能过高，仅用于验证网络的局部信息的有效性  
    def __init__(self,walk_steps:int=10):         
        super().__init__(walk_steps)        
        self.walk_steps = walk_steps
    def train(                                  
        self,
        edge_matrix:Matrix,
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]      # 此为关联矩阵
            edge_size_inv_vector , node_degree_inv_vector = (1 / pos_matrix.sum(axis=0)) , (1 / pos_matrix.sum(axis=1))
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ np.diag(edge_size_inv_vector) @ pos_matrix.T @ np.diag(node_degree_inv_vector)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()            
        multi_steps_sum_matrix = transition_matrix
        for _ in range(self.walk_steps - 1):
            transition_matrix = transition_matrix @ transition_matrix
            multi_steps_sum_matrix += transition_matrix
        self.multi_steps_transition_matrix = multi_steps_sum_matrix
        # 得到预测分数
        return  self(edge_matrix)