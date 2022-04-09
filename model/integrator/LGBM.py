import scipy.sparse as ssp
import numpy as np
from lightgbm import LGBMClassifier
from sklearn import linear_model
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any
)
from .Base import Integrator
class LightGBM(Integrator):
    def __init__(
        self,
        feature_classes = ["NNAA","NEAA","ENAA","EEAA"],
        feature_params:List[Dict[str,Any]] = [{}]*4,
        random_state=None,
        n_jobs= -1,
        verbose =  -1,
        preprocess=None
        ):
        hyper_params = {                     # 该参数字典作为线性回归模型的参数
            "random_state" : random_state,
            "objective"       : "binary",
            "n_jobs"        : n_jobs,
            "verbose"       : verbose
        }

        super().__init__(
            LGBMClassifier,
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
            "n_estimators": range(100,1000,1),
            "max_depth": range(3,8,1),
            "num_leaves" :  range(5, 100, 5),
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": list(np.linspace(0, 1)),
            "reg_lambda": list(np.linspace(0, 1))
        }
        return super().train(edge_matrix,obvious_edge_index,params_space)

    def getFeatureImportance(self):
        self.feature_importances_ = self.model.feature_importances_
        return super().getFeatureImportance()
    
    