import numpy as np
import scipy.sparse as ssp
from itertools import permutations
from model.indicator.Base import Indicator
from .LRW import LocalRandomWalk
from ..module import Matrix
from .SLRW import SuperposedLocalRandomWalk
from typing import (
    Union,
    Optional,
    List
)

class SuperposedLocalTargetRandomWalk(SuperposedLocalRandomWalk): # 此方法时间复杂度可能过高，仅用于验证网络的局部信息的有效性  
    def __init__(self,restart_prob:float=0.2,walk_steps:int=10):         
        super().__init__(walk_steps)        
        assert 1 > restart_prob >= 0 , f"重启概率{restart_prob}不合法！！！合法区间：[0,1)"
        self.restart_prob = restart_prob
        self.walk_steps = walk_steps            # 该数取大于等于1的整数
    def train(                                  
        self,
        edge_matrix:Matrix,
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        if self.restart_prob == 0:
            return super().train(edge_matrix,obvious_edge_index)
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]      # 此为关联矩阵
            edge_size_inv_vector , node_degree_inv_vector = (1 / pos_matrix.sum(axis=0)) , (1 / pos_matrix.sum(axis=1))
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            self.transition_matrix =  pos_matrix @ np.diag(edge_size_inv_vector) @ pos_matrix.T @ np.diag(node_degree_inv_vector)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            edge_size_inv_vector , node_degree_inv_vector = np.asarray(1 / pos_matrix.sum(axis=0)).squeeze() , np.asarray(1 / pos_matrix.sum(axis=1)).squeeze()
            edge_size_inv_vector[np.isinf(edge_size_inv_vector)] = 0
            node_degree_inv_vector[np.isinf(node_degree_inv_vector)] = 0
            self.transition_matrix =  pos_matrix @ ssp.diags(edge_size_inv_vector).tocsr() @ pos_matrix.T @ ssp.diags(node_degree_inv_vector).tocsr()            
        # 得到预测分数
        return  self(edge_matrix)

    def forward(self,edge_matrix:Matrix)->List[float]:
        if self.restart_prob == 0:
            return super().forward(edge_matrix)
        prediction = []
        for edge_index in range(edge_matrix.shape[1]):
            edge_vector = edge_matrix[:,edge_index]
            if isinstance(self.transition_matrix,ssp.spmatrix):
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
            if isinstance(self.transition_matrix,ssp.spmatrix):
                restart_matrix = ssp.csr_matrix(restart_matrix)
            state_matrix = restart_matrix
            scores_log = []
            for _ in range(self.walk_steps):
                state_matrix = self.oneStep(state_matrix,restart_matrix)
                scores_log.append(state_matrix[t_coo[0],t_coo[1]])
            prediction.append(
                np.array(scores_log).sum(axis=0).mean()
                )
        return prediction


    def oneStep(self,cur_state:Matrix,init_state:Matrix)->Matrix:
        if len(cur_state.shape) == 1:
            cur_state = cur_state.reshape(-1,1)
        if len(init_state.shape) == 1:
            init_state = init_state.reshape(-1,1)
        if isinstance(self.transition_matrix,ssp.spmatrix):
            cur_state , init_state = ssp.csr_matrix(cur_state) , ssp.csr_matrix(init_state)
        next_state = (1 - self.restart_prob) * self.transition_matrix @ cur_state + self.restart_prob * init_state
        return next_state