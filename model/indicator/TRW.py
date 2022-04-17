import numpy as np
import scipy.sparse as ssp
from itertools import permutations
from .Base import Indicator
from ..module import Matrix
from typing import (
    Union,
    Optional,
    List
)

class TargetRandomWalk(Indicator):
    def __init__(self,restart_prob:float=0.2):         
        super().__init__()        
        if 1 > restart_prob > 0: 
            self.restart_prob = restart_prob
        else:
            self.restart_prob = None
        self.ss_matrix = None        
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Matrix,
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算邻接矩阵
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]      # 此为关联矩阵
            edge_size_inv_vector , node_degree_inv_vector = (1 / pos_matrix.sum(axis=0)) , (1 / pos_matrix.sum(axis=1))
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ np.diag(edge_size_inv_vector) @ pos_matrix.T @ np.diag(node_degree_inv_vector)
            self.ss_matrix = self.restart_prob * np.linalg.inv(np.eye(transition_matrix.shape[0]) - (1 - self.restart_prob) * transition_matrix)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()        
            self.ss_matrix = self.restart_prob * ssp.linalg.inv(ssp.eye(transition_matrix.shape[0]) - (1 - self.restart_prob) * transition_matrix)
        # 得到预测分数
        return  self(edge_matrix)

    def forward(self,edge_matrix:Matrix)->List[float]:
        prediction = []

        for edge_index in range(edge_matrix.shape[1]):
            edge_vector = edge_matrix[:,edge_index]
            if isinstance(self.ss_matrix,ssp.spmatrix):
                edge_vector = edge_vector.toarray().squeeze()
            node_group = np.where(edge_vector != 0)[0]    
            r_coo , t_coo = [[],[]] , [[],[]]
            for col_index , node in enumerate(node_group):
                restarts = set(node_group) - set([node])
                t_coo[0] += [node]
                t_coo[1] += [col_index]
                r_coo[0] += list(restarts)
                r_coo[1] += [col_index] * len(restarts)
            restart_matrix = np.zeros((edge_vector.shape[0],node_group.shape[0]))
            restart_matrix[r_coo[0],r_coo[1]] = 1
            restart_matrix /= restart_matrix.sum(axis=0)
            if isinstance(self.ss_matrix,ssp.spmatrix):
                restart_matrix = ssp.csr_matrix(restart_matrix)
            prediction.append(
                (self.ss_matrix @ restart_matrix)[t_coo[0],t_coo[1]].mean()
                )
        return prediction