import numpy as np
import scipy.sparse as ssp
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from itertools import combinations
import numpy as np
import scipy.sparse as ssp
from numpy.linalg import LinAlgError
from .Base import Indicator
from ..module import Matrix

from typing import (
    Union,
    Optional,
    Tuple
)

class BaseKatzIndex(Indicator):
    def __init__(self,binarization:bool=True):         
        super().__init__()        
        self.scores_matrix = None
        self.candidated_lambdas = [0.001,0.005,0.01,0.1,0.5]   
        self.binarization = binarization    
    def train(                                  # 该方法用来生成分数矩阵
        self,
        edge_matrix:Matrix,
        obvious_edge_index:np.ndarray
        )->Optional[float]:
        super().train(edge_matrix,obvious_edge_index)
        unobserved_edge_index = np.setdiff1d(np.linspace(0,edge_matrix.shape[1]-1,edge_matrix.shape[1],dtype=np.int64),obvious_edge_index)
        _lambda = self.determineParmByCV(
            (edge_matrix[:,obvious_edge_index],edge_matrix[:,unobserved_edge_index])
            )
        pos_matrix = edge_matrix[:,obvious_edge_index]
        # 计算分数矩阵

        if isinstance(pos_matrix,np.ndarray):
            adj_matrix = pos_matrix @ pos_matrix.T
            if self.binarization:
                adj_matrix[adj_matrix!=0] = 1
            identity_matrix =  np.eye(adj_matrix.shape[0])
            self.scores_matrix = np.linalg.pinv(
                        (identity_matrix - _lambda * adj_matrix)
                    ) - identity_matrix

        elif isinstance(pos_matrix,ssp.spmatrix):
            pos_matrix = pos_matrix.tocsr()
            adj_matrix = (pos_matrix @ pos_matrix.T).tolil()
            if self.binarization:
                adj_matrix[adj_matrix!=0] = 1
            identity_matrix = ssp.csr_matrix(np.eye(adj_matrix.shape[0]))
            temp_matrix = identity_matrix - _lambda * adj_matrix
            try:
                self.scores_matrix = ssp.linalg.inv(temp_matrix) - identity_matrix
            except RuntimeError:
                self.scores_matrix = np.linalg.pinv(temp_matrix.toarray()) - identity_matrix

        # 得到预测分数
        return  self(edge_matrix)
    def determineParmByCV(self,train_set:Tuple[Matrix,Matrix],kfold_num = 5)->float:
        pos_matrix , neg_matrix = train_set
        effect_log = []#np.empty((len(self.candidated_lambdas),kfold_num))
        kf = KFold(n_splits=kfold_num)


        for _ , condicated_lambda in enumerate(self.candidated_lambdas):
            effect_log.append([])
            for _ , index_tuple in enumerate(zip(kf.split(pos_matrix.T) , kf.split(neg_matrix.T))):
                pos_index , neg_index = index_tuple
                try:
                    if isinstance(pos_matrix,np.ndarray):
                        _inc_matrix = pos_matrix[:,pos_index[0]]
                        val_set = np.concatenate((pos_matrix[:,pos_index[1]],neg_matrix[:,neg_index[1]]),axis=1)
                        val_label = np.concatenate( (np.ones(len(pos_index[1])) , np.zeros(len(neg_index[1]))))
                        adj_matrix = _inc_matrix @ _inc_matrix.T
                        adj_matrix[np.diag_indices_from(adj_matrix)] = 0
                        if self.binarization:
                            adj_matrix[adj_matrix!=0] = 1    
                        _inc_matrix = pos_matrix[:,pos_index[0]]
                        identity_matrix =  np.eye(adj_matrix.shape[0])
                        katz_matrix = np.linalg.pinv(
                            (identity_matrix - condicated_lambda * adj_matrix)
                        ) - identity_matrix
                    
                    elif isinstance(pos_matrix,ssp.spmatrix):
                        _inc_matrix = pos_matrix[:,pos_index[0]].tocsr()
                        val_set = ssp.hstack((pos_matrix[:,pos_index[1]],neg_matrix[:,neg_index[1]]))   
                        val_label = np.concatenate( (np.ones(len(pos_index[1])) , np.zeros(len(neg_index[1]))))     
                        adj_matrix = _inc_matrix @ _inc_matrix.T
                        adj_matrix = adj_matrix.tolil()
                        adj_matrix[np.diag_indices_from(adj_matrix)] = 0
                        if self.binarization:
                            adj_matrix[adj_matrix!=0] = 1    
                        _inc_matrix = pos_matrix[:,pos_index[0]].tocsr()
                        identity_matrix =  ssp.csr_matrix(np.eye(adj_matrix.shape[0]))
                        temp_matrix = identity_matrix - condicated_lambda * adj_matrix
                        try:
                            katz_matrix = ssp.linalg.inv(temp_matrix) - identity_matrix
                        except RuntimeError:
                            katz_matrix = np.linalg.pinv(temp_matrix.toarray()) - identity_matrix
                except LinAlgError as _:
                    continue
                else:
                    prediction = []
                    for edge_index in range(val_set.shape[1]):
                        edge_vector = (val_set[:,edge_index])
                        if isinstance(val_set,ssp.spmatrix):
                            edge_vector = edge_vector.toarray().squeeze()
                        node_group = np.where(edge_vector != 0)[0]    
                        node_comb = list(combinations(node_group,r=2))            
                        coo = ([],[])         
                        for node_pair in node_comb:
                            coo[0].append(node_pair[0])
                            coo[1].append(node_pair[1])                      
                        prediction.append(
                            katz_matrix[coo].mean()
                        )
                    effect_log[-1].append(
                        roc_auc_score(val_label,prediction)
                    )
        effect_log = np.array(
            [np.array(effect_vector).mean() for effect_vector in effect_log]
        )
        effect_log[np.isnan(effect_log)] = 0
        
        return self.candidated_lambdas[np.argmax(effect_log)]
