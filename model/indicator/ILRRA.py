import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional,
    List
)
from ..integrator import LogisticRegression

class InvLRResourceAllocation(Indicator):
    def __init__(self,steps:int=3):         
        super().__init__()       
        self.lr = LogisticRegression(
            feature_classes = ["SRA" for _ in range(steps)],
            feature_params = {"steps":step for step in range(1,steps)}
        )
        
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        self.lr.train(edge_matrix,obvious_edge_index)
        best_step = np.array(self.lr.getFeatureImportance.values()).argmax()
        self.scores_matrix = self.feature_objects[best_step]
        return  self(edge_matrix)