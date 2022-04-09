from turtle import width
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

class SuperposedHyperEfficiency(Indicator):
    def __init__(self,width:int=1,ratio_type:str="exp",**params):         
        super().__init__()        
        assert width>=1 , f"参数width_min{width}的值不合理！！"
        self.scores_matrix = None        
        self.width = width                      # 此为叠加walk的最大宽度
        self.ratio_type = ratio_type                      
        self.other_params = params
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
            self.cliques = []
            for _width in range(1,self.width+1):
                coexist_matrix[coexist_matrix < _width] = 0
                coexist_matrix[coexist_matrix != 0] = 1
                self.cliques.append(nx.from_numpy_array(coexist_matrix))
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            coexist_matrix = (pos_matrix @ pos_matrix.T).tolil()
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            self.cliques = []
            for _width in range(1,self.width+1):
                coexist_matrix[coexist_matrix < _width] = 0
                coexist_matrix[coexist_matrix != 0] = 1
                self.cliques.append(nx.from_scipy_sparse_matrix(coexist_matrix))
        
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
            edge_effis = []
            for clique in self.cliques:
                edge_effis.append([])
                for node_pair in node_comb:
                    try:
                        effi = 1 / nx.algorithms.shortest_paths.generic.shortest_path_length(
                            clique,
                            source = node_pair[0],
                            target = node_pair[1]
                            )
                        
                    except nx.NetworkXNoPath as _:
                        effi = 0
                    finally:
                        edge_effis[-1].append(effi)
            edge_effis = np.array(edge_effis).mean(axis=1)
            prediction.append(
                edge_effis @ self.getWeightVector()
            )        
        return  prediction
    def getWeightVector(self)->np.ndarray:
        if self.ratio_type == "exp":
            base_number = self.other_params.get("base_number",0.1)
            exponents = np.linspace(0,self.width-1,self.width)
            weight_vector = np.full_like(exponents,base_number) ** exponents
        #elif self.ratio_type == "fact":

        return weight_vector