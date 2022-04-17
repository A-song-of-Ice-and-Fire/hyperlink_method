from pdb import Restart
import numpy as np
import scipy.sparse as ssp
from itertools import permutations
from .Base import Indicator
from ..module import Matrix
from sklearn.metrics import roc_auc_score
from typing import (
    Union,
    Optional,
    List,
    Tuple
)
from sklearn.model_selection import KFold
class RestartRandomWalk(Indicator):
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

        if not self.restart_prob:
            super().train(edge_matrix,obvious_edge_index)
            self.restart_prob = self.determineParmByCV(
                edge_matrix,obvious_edge_index
            )
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
            for col_index , restart in enumerate(node_group):
                targets = set(node_group) - set([restart])
                t_coo[0] += list(targets)
                t_coo[1] += [col_index] * len(targets)
                r_coo[0] += [restart]
                r_coo[1] += [col_index]
            restart_matrix = np.zeros((edge_vector.shape[0],node_group.shape[0]))
            restart_matrix[r_coo[0],r_coo[1]] = 1
            if isinstance(self.ss_matrix,ssp.spmatrix):
                restart_matrix = ssp.csr_matrix(restart_matrix)
            prediction.append(
                (self.ss_matrix @ restart_matrix)[t_coo[0],t_coo[1]].mean()
                )
        return prediction
    def determineParmByCV(self,edge_matrix:Matrix,obvious_edge_index:np.ndarray,kfold_num = 5)->float:
        unobserved_edge_index = np.setdiff1d(np.linspace(0,edge_matrix.shape[1]-1,edge_matrix.shape[1],dtype=np.int64),obvious_edge_index)

        pos_matrix , neg_matrix = edge_matrix[:,obvious_edge_index] , edge_matrix[:,unobserved_edge_index]
        effect_log = []
        kf = KFold(n_splits=kfold_num)
        candidates = [0.01,0.1,0.2,0.3]

        for condicate in candidates:
            effect_log.append([])
            for index_tuple in zip(kf.split(pos_matrix.T) , kf.split(neg_matrix.T)):
                pos_index , neg_index = index_tuple
                val_model = RestartRandomWalk(condicate)
                if isinstance(edge_matrix,np.ndarray):
                    val_train_sample = np.concatenate((pos_matrix[:,pos_index[0]],neg_matrix[:,neg_index[0]]),axis=1)
                    val_test_sample = np.concatenate((pos_matrix[:,pos_index[1]],pos_matrix[:,neg_index[1]]),axis=1) 
                elif isinstance(edge_matrix,ssp.spmatrix):
                    val_train_sample = ssp.hstack((pos_matrix[:,pos_index[0]],neg_matrix[:,neg_index[0]]))
                    val_test_sample = ssp.hstack((pos_matrix[:,pos_index[1]],neg_matrix[:,neg_index[1]]))
                val_test_label = np.concatenate((np.ones(len(pos_index[1])),np.zeros(len(neg_index[1])))) 
                val_model.train(
                    val_train_sample,
                    obvious_edge_index=np.arange(0,pos_index[0].size,1)
                    )
                val_test_prediction = val_model.test(val_test_sample)                
                effect_log[-1].append(
                    roc_auc_score(val_test_label,val_test_prediction)
                )
        effect_log = np.array(
            [np.array(effect_vector).mean() for effect_vector in effect_log]
        )
        effect_log[np.isnan(effect_log)] = 0
        return candidates[np.argmax(effect_log)]