import numpy as np
import scipy.sparse as ssp
from .Base import Indicator
from typing import (
    Union,
    Optional
)

class HyperWalk(Indicator):
    def __init__(self,width:int=1,length:int=1):         
        super().__init__()        
        assert width>=1 , f"参数width{width}的值不合理！！"
        assert length>=1 , f"参数length{length}的值不合理！！"
        self.scores_matrix = None        
        self.width = width
        self.length = length

        
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
            self.scores_matrix = np.linalg.matrix_power(coexist_matrix,self.length)
        elif isinstance(edge_matrix,ssp.spmatrix):
            pos_matrix = edge_matrix[:,obvious_edge_index].tocsr()
            coexist_matrix = (pos_matrix @ pos_matrix.T).tolil()
            coexist_matrix[np.diag_indices_from(coexist_matrix)] = 0
            coexist_matrix[coexist_matrix < self.width] = 0
            coexist_matrix[coexist_matrix != 0] = 1
            coexist_matrix = coexist_matrix.tocsr()
            self.scores_matrix = coexist_matrix ** self.length
        # print(f"width:{self.width},length:{self.length}")
        # 得到预测分数
        return  self(edge_matrix)