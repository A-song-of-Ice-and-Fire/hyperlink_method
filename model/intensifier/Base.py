import numpy as np
import scipy.sparse as ssp
from itertools import combinations
from ..module import Module
from typing import (
    Union,
    Optional,
    List
)
from ..indicator.Base import Indicator
from ..indicator import *
class Intensifier(Module):
    def __init__(self,raw_indicator_abb:str):
        super().__init__()
        self.raw_indicator_abb = raw_indicator_abb
        self.scores_matrix = None
    def train(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train()
        raw_indicator = indicator_abb_map[self.raw_indicator_abb]()
        raw_indicator.train(edge_matrix,obvious_edge_index)
        self.raw_matrix = raw_indicator.scores_matrix
        # 该方法需要重写
        ...
    def test(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        )->Optional[List[float]]:
        # 得到预测分数
        super().test()
        return  self(edge_matrix)
    def forward(self,edge_matrix:Union[np.ndarray,ssp.spmatrix])->Optional[List[float]]:
        prediction = []

        for edge_index in range(edge_matrix.shape[1]):
            edge_vector = (edge_matrix[:,edge_index])
            if isinstance(edge_matrix,ssp.spmatrix):
                edge_vector = edge_vector.toarray().squeeze()
            node_group = np.where(edge_vector != 0)[0]    
            node_comb = list(combinations(node_group,r=2))            
            coo = ([],[])         
            for node_pair in node_comb:
                coo[0].append(node_pair[0])
                coo[1].append(node_pair[1])                      
            prediction.append(
                self.scores_matrix[coo].mean()
            )        
        return  prediction