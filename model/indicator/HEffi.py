import numpy as np
import networkx as nx
import scipy.sparse as ssp
from itertools import combinations
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)

class HyperEfficiency(Indicator):
    def __init__(self,width:int=1):         
        super().__init__()        
        assert width>=1 , f"参数width{width}的值不合理！！"
        self.scores_matrix = None        
        self.width = width
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        # 计算共存矩阵
        if isinstance(edge_matrix,np.ndarray):
            pos_matrix = edge_matrix[:,obvious_edge_index]
            coexist_matrix = pos_matrix @ pos_matrix.T
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            coexist_matrix[coexist_matrix < self.width] = 0
            coexist_matrix[coexist_matrix != 0] = 1
            self.clique = nx.from_numpy_array(coexist_matrix)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            coexist_matrix = (pos_matrix @ pos_matrix.T).tolil()
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            coexist_matrix[coexist_matrix < self.width] = 0
            coexist_matrix[coexist_matrix != 0] = 1
            self.clique = nx.from_scipy_sparse_matrix(coexist_matrix)
        
        # 得到预测分数
        return  self(edge_matrix)
    def forward(self,edge_matrix:Union[np.ndarray,ssp.spmatrix])->Optional[List[float]]:
        prediction = []

        for edge_index in range(edge_matrix.shape[1]):
            edge_vector = (edge_matrix[:,edge_index])
            if isinstance(edge_matrix,ssp.spmatrix):
                edge_vector = edge_vector.toarray().squeeze()
            node_group = np.where(edge_vector != 0)[0]    
            node_comb = list(combinations(node_group,r=2))                    
            nodes_effi = []
            for node_pair in node_comb:
                try:
                    effi = 1 / nx.algorithms.shortest_paths.generic.shortest_path_length(
                        self.clique,
                        source = node_pair[0],
                        target = node_pair[1]
                        )
                    
                except nx.NetworkXNoPath as _:
                    effi = 0
                finally:
                    nodes_effi.append(effi)
            prediction.append(
                np.array(nodes_effi).mean()
            )        
        return  prediction