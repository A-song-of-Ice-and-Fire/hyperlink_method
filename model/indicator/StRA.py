import numpy as np
import scipy.sparse as ssp

from model.indicator.RA import ResourceAllocation
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)
from ..module import Matrix

class StableResourceAllocation(Indicator):
    def __init__(self,restart_prob:int=0.01):         
        super().__init__()        
        assert 1 > restart_prob > 0 , f"重启概率{restart_prob}不合法，合法区间：(0,1)"
        self.restart_prob = restart_prob
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]      # 此为关联矩阵
            edge_size_inv_vector , node_degree_inv_vector = (1 / pos_matrix.sum(axis=0)) , (1 / pos_matrix.sum(axis=1))
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ np.diag(edge_size_inv_vector) @ pos_matrix.T @ np.diag(node_degree_inv_vector)
            ss_matrix = (self.restart_prob * np.linalg.inv(np.eye(transition_matrix.shape[0]) - (1 - self.restart_prob) * transition_matrix)).T
            node_degree_vector = pos_matrix.sum(axis=1)
            self.scores_matrix = np.diag(node_degree_vector) @ ss_matrix
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()
            ss_matrix = (self.restart_prob * ssp.linalg.inv(ssp.eye(transition_matrix.shape[0]) - (1 - self.restart_prob) * transition_matrix)).T
            node_degree_vector = np.asarray(pos_matrix.sum(axis=1)).squeeze()
            self.scores_matrix = ssp.diags(node_degree_vector) @ ss_matrix 
        # 得到预测分数
        return  self(edge_matrix)