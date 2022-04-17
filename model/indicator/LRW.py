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

class LocalRandomWalk(Indicator): # 此方法时间复杂度可能过高，仅用于验证网络的局部信息的有效性  
    def __init__(self,steps:int=10):         
        super().__init__()        
        self.steps = steps
        self.multi_steps_transition_matrix = None           # 该方法用来生成分数矩阵
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
            self.multi_steps_transition_matrix = np.linalg.matrix_power(transition_matrix,self.steps)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()            
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()            
            self.multi_steps_transition_matrix = transition_matrix**self.steps
        # 得到预测分数
        return  self(edge_matrix)

    def forward(self,edge_matrix:Matrix)->List[float]:
        prediction = []
        for edge_index in range(edge_matrix.shape[1]):
            edge_vector = edge_matrix[:,edge_index]
            if isinstance(self.multi_steps_transition_matrix,ssp.spmatrix):
                edge_vector = edge_vector.toarray().squeeze()
            node_group = np.where(edge_vector != 0)[0] 
            t_coo , i_coo = [[],[]] , [[],[]]
            for col_index , node in enumerate(node_group):
                init_nodes = set(node_group) - set([node])
                t_coo[0] += [node]
                t_coo[1] += [col_index]
                i_coo[0] += list(init_nodes)
                i_coo[1] += [col_index] * len(init_nodes)
            init_matrix = np.zeros((edge_vector.shape[0],node_group.shape[0]))
            init_matrix[i_coo[0],i_coo[1]] = 1
            init_matrix /= init_matrix.sum(axis=0)
            if isinstance(self.multi_steps_transition_matrix,ssp.spmatrix):
                init_matrix = ssp.csr_matrix(init_matrix)
            state_matrix = self.multi_steps_transition_matrix @ init_matrix
            prediction.append(
                state_matrix[t_coo[0],t_coo[1]].mean()
                )
        return prediction