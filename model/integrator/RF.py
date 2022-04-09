import scipy.sparse as ssp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any
)
from .Base import Integrator
class RandomForest(Integrator):
    def __init__(
        self,
        feature_classes = ["NNAA","NEAA","ENAA","EEAA"],
        feature_params:List[Dict[str,Any]] = [{}]*4,
        random_state=None,
        n_jobs=-1,
        verbose = 0,
        preprocess=None
        ):
        hyper_params = {                     # 该参数字典作为线性回归模型的参数
            "random_state" : random_state,
            "n_jobs"        : n_jobs,
            "verbose"       : verbose
        }

        super().__init__(
            RandomForestClassifier,
            feature_classes,
            feature_params,
            preprocess,
            hyper_params
            )

    def train(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray
        )->Optional[float]:

        params_space = {
                "n_estimators" : range(10,71),
                "max_depth":range(1,21)
            }
        return super().train(edge_matrix,obvious_edge_index,params_space)

    def getFeatureImportance(self):
        self.feature_importances_ = self.model.feature_importances_
        return super().getFeatureImportance()
    
    