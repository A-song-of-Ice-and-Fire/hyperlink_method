from modulefinder import Module
import numpy as np
import os
import scipy.sparse as ssp
import datetime
import pandas as pd
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

def downSampleByMNS(inc_matrix:Union[np.ndarray,ssp.spmatrix],sample_times:int)->np.ndarray: # MNS下采样
    
    positive_set = set()    #正样本信息

    
    # 将关联矩阵转化为超边集合
    for edge_index in range(inc_matrix.shape[1]):
        if isinstance(inc_matrix,ssp.spmatrix):
            edge_vector = np.asarray(inc_matrix.tocsc()[:,edge_index].todense()).squeeze()
        elif isinstance(inc_matrix,np.ndarray):
            edge_vector = inc_matrix[:,edge_index]         
        positive_set.add(
            frozenset(np.where(edge_vector != 0)[0])
            )
    
    # 得到团图的信息
    if isinstance(inc_matrix,ssp.spmatrix):
        inc_matrix = inc_matrix.tocsr()
        clique_adj = (inc_matrix @ inc_matrix.T).toarray()
        
    else:      
        clique_adj = inc_matrix @ inc_matrix.T  
        clique_adj[np.diag_indices_from(clique_adj)] = 0
    clique_adj[clique_adj != 0] = 1
    adj_table = {start_point :  set(np.where(clique_adj[start_point,:] !=0 )[0]) for start_point in range(clique_adj.shape[0])}
    # 将邻接信息转化为邻接表，以字典方式存储
    

    # positive_set中包含了所有已存在的超边
    candidate_base , base_dist = getBD(inc_matrix,2)
    negative_set = set()

    with alive_bar(sample_times,title="Negative sampling") as bar:
        while len(negative_set) < sample_times:
            base = np.random.choice(candidate_base,size=1,p=base_dist)[0]
            # 维护一个连通成员集合init_members，该集合内所有成员必然有base个连通成员
            init_members = np.where(clique_adj.sum(axis=0) >= base)[0]

            # 维护两个集合，members是已经被选中的成员集合，这些成员最终构成了负边，candidates是候选者集合，该集合内任一成员与members内的所有成员均连通
            members = set()
            candicates = set()
            init_member = np.random.choice(init_members,size=1)[0]
            members.add(init_member)
            candicates.update(adj_table[init_member])
            while len(members) < base:
                new_member = candicates.pop()
                members.add(new_member)
                candicates.update(adj_table[new_member])

            
            negative_edge = frozenset(
                members
            )

            if (negative_edge in positive_set) or (negative_edge in negative_set):
                continue
            negative_set.add(
                negative_edge
            )
            bar()

    # 将抽样得到的负边集合转化为矩阵形式
    row = []
    col = []
    for edge_index , node_indice in enumerate(negative_set):        
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
def evalution(label:Iterable,prediction:Iterable,metrics:str="all",**kwargs)->Dict[str,Union[float,Dict[str,float]]]:
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
        res["PRECISION@N"] = {}
        # for threshold_num in threshold_nums:
        #     binary_pred = np.zeros_like(label)
        #     binary_pred[np.argsort(-prediction)[:threshold_num]] = 1
        #     precision = precision_score(label,binary_pred)
        #     res["PRECISION@N"][str(threshold_num)] = precision

        # 对df_temp首先进行随机打乱，再基于prediction列进行稳定排序，相当于在依照prediction列的值进行排序的过程中，对相同的值作随机排序
        df_temp = pd.DataFrame({"label":label,"prediction":prediction})
        df_temp = df_temp.sample(frac=1,random_state=np.random.randint(0,high=1e4),ignore_index=True)
        df_temp = df_temp.sort_values(by="prediction",ascending=False,axis="index",kind = "stable",ignore_index=True)
        for threshold_num in range(1,df_temp.shape[0]+1):
            pos_vector = df_temp.loc[:,"label"].iloc[:threshold_num].values
            res["PRECISION@N"][str(threshold_num)] = pos_vector.sum() / pos_vector.size
    return res
    
def evalutionAsBase(base_vector:np.ndarray,label:Iterable,prediction:Iterable,joint_assess_lb:int=10)->Dict[str,Optional[Dict[str,float]]]:
    eval_dict = {}
    label = np.array(label)
    prediction = np.array(prediction)
    for base in range(2,joint_assess_lb+1):
        if base < joint_assess_lb:
            index_with_base = np.where(base_vector == base)
            base_str = f"{base}"
        else:
            index_with_base = np.where(base_vector >= base)
            base_str = f"{base}+"
        label_with_base , prediction_with_base = label[index_with_base] , prediction[index_with_base]
        if label_with_base.sum() == len(label_with_base) or label_with_base.sum() == 0:     # 全是正样本或负样本时，没有计算的意义
            eval_dict[str(base)] = None
            continue
        eval_dict[base_str] = evalution(label_with_base,prediction_with_base)
    return eval_dict
def saveLog(df_prediction:DataFrame,df_metric:Dict[str,DataFrame],save_dir:str,model:Module,model_params:Dict[str,Any],timestamp:str)->None:
        save_metric_dir = os.path.join(save_dir,"metric_score")
        save_prediction_dir = os.path.join(save_dir,"prediction_score")
        
        if not os.path.exists(save_metric_dir):
            os.makedirs(save_metric_dir)
        if not os.path.exists(save_prediction_dir):
            os.makedirs(save_prediction_dir)
        model_str = object_to_str(model)
        if isinstance(model,Intensifier):
            model_str = f"{model_str}-{model_params['raw_indicator_abb']}"
        elif isinstance(model,Indicator):
            for value in model_params.values():
                model_str += f"-{value}"
        for key in df_metric.keys():
            if df_metric.get(key,None) is not None:
                df_metric[key].to_csv(os.path.join(save_metric_dir,f"{model_str}_{key}_{timestamp}.csv"))
        df_prediction.to_csv(os.path.join(save_prediction_dir,f"{model_str}_{timestamp}.csv"))
        