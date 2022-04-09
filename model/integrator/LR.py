import scipy.sparse as ssp
import numpy as np
from scipy.stats import uniform
from sklearn import linear_model
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any
)
from .Base import Integrator
class LogisticRegression(Integrator):
    def __init__(
        self,
        feature_classes = ["NNAA","NEAA","ENAA","EEAA"],
        feature_params:List[Dict[str,Any]] = [{}]*4,
        solver:str="lbfgs",
        random_state=None,
        max_iter = 500,
        verbose=0,
        n_jobs=-1,
        preprocess="z",
        ):
        """
        solver：优化器，可选：liblinear（坐标下降法）、lbfgs（拟牛顿法）、sag（随机梯度下降法）
        random_state：随机数种子，
        max_iter：优化的最大迭代次数，
        verbose：日志冗长度：0（不输出训练过程）、1（偶尔输出）、 >1（对每个子模型都输出），
        n_jobs：并行数，-1（跟CPU核数一致）、1（默认值），
        preprocess：特征处理方式，None（不处理）、z（标准化）、m（归一化）、n（行单位向量化）
        """
        hyper_params = {                     # 该参数字典作为线性回归模型的参数
            "solver" : solver,
            "random_state" : random_state,
            "max_iter"      : max_iter,
            "verbose"       : verbose,
            "n_jobs"        : n_jobs
        }

        super().__init__(
            linear_model.LogisticRegression,
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
                "C" : uniform(loc=0, scale=4),
                "penalty" :['l2', 'l1']
            }
        return super().train(edge_matrix,obvious_edge_index,params_space)

    def getFeatureImportance(self):
        self.feature_importances_ = self.model.coef_[0]
        return super().getFeatureImportance()
    
    