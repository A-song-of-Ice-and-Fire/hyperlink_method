import scipy.sparse as ssp
import numpy as np
import networkx as nx
from itertools import combinations
from ..module import Module
from lightgbm import LGBMClassifier
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer
)
from typing import (
    Any,
    Iterable,
    Union,
    Optional,
    List,
    Dict
)
from ..indicator import indicator_abb_map

class Integrator(Module):
    def __init__(
        self,
        classifier:ClassifierMixin,
        feature_classes:List[str] = ["NNAA","NEAA","ENAA","EEAA"],
        feature_params:List[Dict[str,Any]] = [{}]*4,
        preprocess:Optional[str] = None,
        hyper_params:Dict[str,Any] = {}
        ):
        super().__init__()
        self._hyper_params = hyper_params
        self._classifier = classifier
        self.model = self._classifier(**self._hyper_params)
        
        assert len(feature_classes) == len(feature_params) , \
            f"特征个数：{len(feature_classes)}与特征参数数量{len(feature_params)}不匹配！！！！"

        self.feature_classes = feature_classes
        self.feature_params = feature_params
        if preprocess == "z":
            self._scaler = StandardScaler()
        elif preprocess == "m":
            self._scaler = MinMaxScaler()
        elif preprocess == "n":
            self._scaler = Normalizer()
        else:
            self._scaler = None

        self.feature_importances_:Optional[Iterable] = None
    def train(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        obvious_edge_index:np.ndarray,
        params_space:Dict[str,Any]
        )->Optional[float]:
        super().train()
        
        # 得到训练后的特征对象集合
        self.feature_objects = []
        for feature_class,feature_param in zip(self.feature_classes,self.feature_params):
            feature_object = indicator_abb_map.get(feature_class,None)(**feature_param)
            feature_object.train(edge_matrix,obvious_edge_index)
            self.feature_objects.append(feature_object)


        X_train = self.getFeatureVector(edge_matrix)
        searcher = RandomizedSearchCV(
            self.model, params_space, n_iter=300, scoring="roc_auc",n_jobs=-1
        )
        if self._scaler:
            X_train = self._scaler.fit_transform(X_train)
        label_train = np.zeros(edge_matrix.shape[-1])
        label_train[obvious_edge_index] = 1
        searcher.fit(X_train,label_train)
        self.model = self._classifier(
                **{**searcher.best_params_ , **self._hyper_params}
            )
        self.model.fit(X_train,label_train)
        
        return self.model.predict_proba(X_train)[:,1]
    
    
    
    def test(
        self,
        edge_matrix:Union[np.ndarray,ssp.spmatrix],
        )->Optional[float]:
        super().test()
        return self(edge_matrix)
    def forward(self,edge_matrix:Union[np.ndarray,ssp.spmatrix])->Optional[float]:
        X = self.getFeatureVector(edge_matrix)
        if self._scaler:
            X = self._scaler.transform(X)
        return self.model.predict_proba(X)[:,1]
    def getFeatureVector(self,edge_matrix:Union[np.ndarray,ssp.spmatrix])->np.ndarray:
        X = np.array(
            [feature_object(edge_matrix) for feature_object in self.feature_objects]
        ).T
        return X
    def getFeatureImportance(self):
        fi_dict = {}
        for index in range(len(self.feature_classes)):
            feature_str , importance = self.feature_classes[index] , self.feature_importances_[index]
            for value in self.feature_params[index].values():
                feature_str += f"-{value}"
            fi_dict[feature_str] = importance
        return fi_dict
        # {feature_str:importance for feature_str,importance in zip(self.feature_classes,self.model.feature_importances_)}
        # {feature_str:importance for feature_str,importance in zip(self.feature_classes,self.model.coef_[0])}
        # {feature_str:importance for feature_str,importance in zip(self.feature_classes,self.model.feature_importances_)}