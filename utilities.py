from modulefinder import Module
import numpy as np
import os
import scipy.sparse as ssp
import datetime
from alive_progress import alive_bar
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score
)
from pandas import DataFrame
from model import object_to_str
from argparse import Namespace
from typing import (
    Any,
    Iterable,
    Optional,
    Tuple,
    Union,
    Dict
)
from model.indicator.Base import Indicator
from model.intensifier.Base import Intensifier
from model.integrator.Base import Integrator
# getBD
# 输入：超图关联矩阵，基数下界
# 输出：大于等于基数下界的所有基数，对应的基数分布
def getBD(inc_matrix:Union[np.ndarray,ssp.spmatrix],lower_bound:int=2)->Tuple[np.ndarray,np.ndarray]:
        edge_size_vector = np.asarray(inc_matrix.sum(axis=0)).squeeze().astype(np.int64)
        max_base = edge_size_vector.max()
        assert max_base >= lower_bound , f"基数下界{lower_bound}超过了最大基数{max_base}！！！！！"
        dist_vector = np.zeros(max_base-lower_bound+1)
        for base in range(lower_bound,max_base+1):
            dist_vector[base-lower_bound] = (edge_size_vector == base).sum()
        return np.linspace(lower_bound,max_base,max_base-lower_bound+1,dtype=np.int64) , dist_vector / dist_vector.sum() 



# downSampleForNE（Negative Edge）
# 输入：超图关联矩阵，取样负边个数
# 输出：负边关联矩阵
def downSampleForNE(inc_matrix:Union[np.ndarray,ssp.spmatrix],sample_times:int)->np.ndarray:
    
    edge_set = set()
    for edge_index in range(inc_matrix.shape[1]):
        if isinstance(inc_matrix,ssp.spmatrix):
            edge_vector = np.asarray(inc_matrix.tocsc()[:,edge_index].todense()).squeeze()
        elif isinstance(inc_matrix,np.ndarray):
            edge_vector = inc_matrix[:,edge_index]         
        edge_set.add(
            frozenset(np.where(edge_vector != 0)[0])
            )

    # edge_set中包含了所有已存在的超边
    
    candidate_base , base_dist = getBD(inc_matrix,2)
    negative_edge_set = set()
    with alive_bar(sample_times,title="Negative sampling") as bar:
        counter = 0
        while counter < sample_times:
        
            base = np.random.choice(candidate_base,size=1,p=base_dist)[0]
            negative_edge = frozenset(
                np.random.choice(
                    np.linspace(0,inc_matrix.shape[0]-1,inc_matrix.shape[0],dtype=np.int32),
                    size = base,
                    replace=False
                )
            )
            if (negative_edge in edge_set) or (negative_edge in negative_edge_set):
                continue
        
            negative_edge_set.add(
                negative_edge
            )
            counter += 1
            bar()
    # 将抽样得到的负边集合转化为矩阵形式
    row = []
    col = []
    for edge_index , node_indice in enumerate(negative_edge_set):        
        row += list(node_indice)
        col += [edge_index] * len(node_indice)

    data = np.ones_like(row)
    NE_matrix = ssp.coo_matrix(
        (data,(row,col)),
        shape = (inc_matrix.shape[0],max(col)+1)
    )

    if isinstance(inc_matrix,np.ndarray):
        NE_matrix = NE_matrix.toarray()
    return NE_matrix

# metrics可选：AUROC、AUPR、all
def evalution(label:Iterable,prediction:Iterable,metrics:str="all",**kwargs)->Tuple:
    assert metrics in ["AUROC","AUPR","PRECISION","PRECISION@N","all"] , f"metrics可选：AUROC、AUPR、PRECISION、PRECISION@N、all，而非{metrics}"
    label , prediction = np.array(label) , np.array(prediction)

    res = {}
    if metrics in ["AUROC","all"]:
        auroc = roc_auc_score(label,prediction)
        res["AUROC"] = auroc
    if metrics in ["AUPR","all"]:
        precision , recall , _ = precision_recall_curve(label,prediction)
        aupr = auc(recall,precision)
        res["AUPR"] = aupr
    if metrics in ["PRECISION","all"]:
        threshold_num = (label == 1).sum()
        binary_pred = np.zeros_like(label)
        binary_pred[np.argsort(-prediction)[:threshold_num]] = 1
        precision = precision_score(label,binary_pred)
        res["PRECISION"] = precision
    if metrics in ["PRECISION@N","all"]:
        max_threshold_num = min(kwargs.get(threshold_num,50),prediction.size)
        threshold_nums = np.arange(5,max_threshold_num+1,5)
        res["PRECISION@N"] = {}
        for threshold_num in threshold_nums:
            binary_pred = np.zeros_like(label)
            binary_pred[np.argsort(-prediction)[:threshold_num]] = 1
            precision = precision_score(label,binary_pred)
            res["PRECISION@N"][str(threshold_num)] = precision
    return res
    
def evalutionAsBase(edge_matrix:np.ndarray,label:Iterable,prediction:Iterable,joint_assess_lb:int=10)->Dict[str,Optional[Dict[str,float]]]:
    eval_dict = {}
    label = np.array(label)
    prediction = np.array(prediction)

    base_vector = np.asarray(edge_matrix.sum(axis=0)).squeeze()
    for base in range(2,joint_assess_lb+1):
        if base < joint_assess_lb:
            index_with_base = np.where(base_vector == base)
            base_str = f"{base}"
        else:
            index_with_base = np.where(base_vector >= base)
            base_str = f"{base}+"
        label_with_base = label[index_with_base]
        if label_with_base.sum() == len(label_with_base) or label_with_base.sum() == 0:
            eval_dict[str(base)] = None
            continue
        prediction_with_base = prediction[index_with_base]
        auroc = roc_auc_score(label_with_base,prediction_with_base)
        precision , recall , _ = precision_recall_curve(label_with_base,prediction_with_base)
        aupr = auc(recall,precision)
        
        # 计算precision（以正样本数作为阈值数量）
        threshold_num = (label_with_base == 1).sum()
        binary_pred = np.zeros_like(label_with_base)
        binary_pred[np.argsort(-prediction_with_base)[:threshold_num]] = 1
        precision = precision_score(label_with_base,binary_pred)


        eval_dict[base_str] = {}
        eval_dict[base_str]["PRECISION"] = precision
        eval_dict[base_str]["AUROC"] = auroc
        eval_dict[base_str]["AUPR"] = aupr                
    return eval_dict
def saveLog(df_dict:Dict[str,DataFrame],save_dir:str,model:Module,model_params:Dict[str,Any],timestamp:str)->None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_str = object_to_str(model)
        if isinstance(model,Intensifier):
            model_str = f"{model_str}-{model_params['raw_indicator_abb']}"
        elif isinstance(model,Indicator):
            for value in model_params.values():
                model_str += f"-{value}"
        for key in df_dict.keys():
            if df_dict.get(key,None) is not None:
                df_dict[key].to_csv(os.path.join(save_dir,f"{model_str}_{key}_{timestamp}.csv"))