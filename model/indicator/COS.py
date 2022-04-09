import numpy as np
import scipy.sparse as ssp
from itertools import combinations
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)
class CosineSimilarity(Indicator):
    def __init__(self):
        super().__init__()
    def train(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:


        # 计算邻接矩阵
        if isinstance(edge_matrix,np.ndarray):
            self.pos_matrix = edge_matrix[:,obvious_edge_index]
        elif isinstance(edge_matrix,ssp.spmatrix):
            self.pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()

        return self(edge_matrix)

        

    def forward(self, edge_matrix: Union[np.ndarray, ssp.spmatrix]) -> Optional[List[float]]:
        # 得到预测分数
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
            left_node_matrix = self.pos_matrix[coo[0],:]
            right_node_matrix = self.pos_matrix[coo[1],:]
            if isinstance(edge_matrix,ssp.spmatrix):
                left_node_matrix = left_node_matrix.toarray()
                right_node_matrix = right_node_matrix.toarray()
            dot_vector = (left_node_matrix * right_node_matrix).sum(axis=1)
            reg_vector = np.linalg.norm(left_node_matrix,axis=1) * np.linalg.norm(right_node_matrix,axis=1)
            cos_vector = dot_vector / reg_vector
            cos_vector[np.isnan(cos_vector)]=0
            prediction.append(
                cos_vector.mean()
            )              
        return  prediction