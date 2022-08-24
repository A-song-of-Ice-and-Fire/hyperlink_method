import time
import numpy as np
import random
import scipy.sparse as ssp
import os
import datetime
import warnings
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from _parser import (
    save_dir,
    dataset_dir,
    dataset_names,
    JE_start_point,
    str_to_object,
    kfold_num,
    random_seed,
    cmd_args,
    processingCommandParam
)
from utilities import (
    downSampleByMNS,
    evalution,
    evalutionAsBase,
    saveLog, 
    downSampleByMNS
)
from model.integrator.Base import Integrator
from model.intensifier.Base import Intensifier
from model import (
    HyperWalk,
    HyperEfficiency
)

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 处理命令行参数
    res_dict = processingCommandParam(cmd_args)
    model_str , repeat_num , is_interator , model_params , dataset_names , is_save , evalution_params , ns_num = \
        res_dict["model_str"] , res_dict["repeat_num"] , res_dict["is_interator"] , res_dict["model_params"] , res_dict["dataset_names"] , res_dict["is_save"] , res_dict["evalution_params"] , res_dict["ns_num"]
        

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # 第一重循环：遍历所设置的数据集
    for dataset_name in dataset_names:              
        for inc_matrix_file in os.listdir(os.path.join(dataset_dir,dataset_name)):
            if inc_matrix_file.split(".")[-1] in ["npy","npz"]:
                break
        else:
            raise IOError(f"数据集 {dataset_name}的邻接矩阵文件未找到！！！")
        inc_path = os.path.join(dataset_dir,dataset_name,inc_matrix_file)
        suffix = os.path.splitext(inc_path)[-1][1:]

        if suffix == "npy":
            inc_matrix = np.load(inc_path).astype(np.float32)
        elif suffix == "npz":
            inc_matrix = ssp.load_npz(inc_path).tocsc().astype(np.float32)

        # 筛选掉大小为1的超边
        edge_size_vector = np.asarray(inc_matrix.sum(axis=0)).squeeze()
        PO_matrix = inc_matrix[:, (edge_size_vector > 1)]

        # 用来记录每一次实验的评价指标
        columns_str = [str(x) for x in range(2,JE_start_point.get(dataset_name,11)+1)]
        columns_str[-1] += "+"
        columns_str.append("all")
        df_metric = {}
        df_metric["AUROC"],df_metric["AUPR"],df_metric["PRECISION"] = \
            pd.DataFrame(columns=columns_str) , pd.DataFrame(columns=columns_str) , pd.DataFrame(columns=columns_str)
        columns_str_threshold = [str(x) for x in np.arange(5,evalution_params["precision_threshold"]+1,5)]
        df_metric["PRECISION@N"] = pd.DataFrame(columns=["edge_size","experiment_No",*columns_str_threshold])
        # 记录每一次实验的特征重要性
        if is_interator:
            feature_strs = []
            for  feature_class,feature_params in zip(model_params["feature_classes"],model_params["feature_params"]):
                feature_strs.append(
                    "-".join([feature_class] + list(map(str,feature_params.values())))
                )
            df_metric["FI"] = pd.DataFrame(columns=feature_strs)

        # 记录实验的预测结果
        df_prediction = pd.DataFrame( columns = ["experiment_No","fold_No","edge_size","label","score"] )

        
        # 第二重循环：重复做repeat_num次实验
        for repeat_n in range(1,repeat_num+1):
            print(f"\n\n\n{'dataset_name':<15s}:{dataset_name}\n{'model_str':<15s}:{model_str}\n{'repeat_n':<15s}:{repeat_n}/{repeat_num}\n")
            # 构造负样本（依照初始的超边基数分布）,形式为负边矩阵
            if ns_num<=0:
                ns_num = PO_matrix.shape[1]
                
            NE_matrix = downSampleByMNS(PO_matrix,ns_num)
            if suffix == "npz":
                NE_matrix = NE_matrix.tocsc()        
            kf = KFold(n_splits=kfold_num)

            # 下述两个列表用来记录五折交叉验证的指标,其中训练集不分基数记录，但测试集分基数记录

            log_dict = {}
            log_dict["train"] = {
                "AUROC" : [],
                "AUPR" : [],
                "PRECISION" : [],
                "PRECISION@N" : {x : [] for x in columns_str_threshold}
            }
            log_dict["test"] = {
                "AUROC" : {x : [] for x in columns_str},
                "AUPR" : {x : [] for x in columns_str},
                "PRECISION" : {x : [] for x in columns_str},
                "PRECISION@N" : {x : {y: []for y in columns_str_threshold} for x in columns_str}
            }
            
            # 记录每一次实验的特征重要性
            if is_interator:
                log_dict["FI"] = defaultdict(list)
                #fi_log = {_class:[] for _class in model_params["feature_classes"]}
            
            # 第三重循环：进行五折交叉验证
            for kfold_counter , index_tuple in enumerate(zip(kf.split(PO_matrix.T) , kf.split(NE_matrix.T))):
                start_time = time.time()
                
                pos_index , neg_index = index_tuple
                if suffix == "npy":
                    train_sample = np.concatenate((PO_matrix[:,pos_index[0]],NE_matrix[:,neg_index[0]]),axis=1)
                    test_sample = np.concatenate((PO_matrix[:,pos_index[1]],NE_matrix[:,neg_index[1]]),axis=1) 
                elif suffix == "npz":
                    train_sample = ssp.hstack((PO_matrix[:,pos_index[0]],NE_matrix[:,neg_index[0]]))
                    test_sample = ssp.hstack((PO_matrix[:,pos_index[1]],NE_matrix[:,neg_index[1]]))
                train_label = np.concatenate((np.ones(len(pos_index[0])),np.zeros(len(neg_index[0]))))
                test_label = np.concatenate((np.ones(len(pos_index[1])),np.zeros(len(neg_index[1])))) 
                
                model = str_to_object[model_str](**model_params)


                train_prediction = model.train(
                    train_sample,
                    np.linspace(0,len(pos_index[0])-1,len(pos_index[0]),dtype=np.int64)
                    )            
                train_result = evalution(train_label,train_prediction,**evalution_params)
                test_prediction = model.test(test_sample)
                test_result = evalution(test_label,test_prediction,**evalution_params)   
                test_result_b = evalutionAsBase(np.asarray(test_sample.sum(axis=0)).squeeze(),test_label,test_prediction,JE_start_point.get(dataset_name,11))                
                
                end_time = time.time()

                # 记录每次实验的预测值
                df_prediction = pd.concat(
                    [
                        df_prediction,
                        pd.DataFrame({
                            "experiment_No": [repeat_n-1]*test_label.size,
                            "fold_No"      : [kfold_counter]*test_label.size,
                            "edge_size"    : np.asarray(test_sample.sum(axis=0)).astype(np.int).squeeze(),
                            "label"        : test_label.astype(np.int),
                            "score"        : test_prediction
                            })
                        ],
                    axis="index",
                    ignore_index=True

                )




                print_str = f"\nKflod[{kfold_counter+1}/{kfold_num}] in {end_time-start_time:.1f}s:\n" + \
                            f"    {'train':5s} AUROC:{train_result.get('AUROC',None):.3f},AUPR:{train_result.get('AUPR',None):.3f},PRECISION:{train_result.get('PRECISION',None):.3f}\n" + \
                            f"    {'test':5s} AUROC:{test_result.get('AUROC',None):.3f},AUPR:{test_result.get('AUPR',None):.3f},PRECISION:{test_result.get('PRECISION',None):.3f}"
                            
                print(print_str)    

                for key in test_result_b.keys():
                    if test_result_b[key]:
                        log_dict["test"]["AUROC"][key].append(test_result_b[key]["AUROC"])
                        log_dict["test"]["AUPR"][key].append(test_result_b[key]["AUPR"])
                        log_dict["test"]["PRECISION"][key].append(test_result_b[key]["PRECISION"])
                        for _threshold in log_dict["test"]["PRECISION@N"][key].keys():
                            log_dict["test"]["PRECISION@N"][key][_threshold].append(test_result_b[key]["PRECISION@N"].get(_threshold,None))
                for key in train_result.keys():
                    if key == "PRECISION@N":
                        for _key in log_dict["train"][key].keys():
                            log_dict["train"][key][_key].append(train_result[key].get(_key,None))
                            log_dict["test"][key]["all"][_key].append(test_result[key].get(_key,None))
                    else:
                        log_dict["train"][key].append(train_result.get(key,None))
                        log_dict["test"][key]["all"].append(test_result.get(key,None))
                
                # 获得特征重要性
                if is_interator:
                    for key,value in model.getFeatureImportance().items():  #这里需要修改
                        log_dict["FI"][key].append(value)


            # 记录每一次实验的评价值
            
            for key in df_metric.keys():
                if key == "FI" and is_interator:
                    df_metric["FI"] = df_metric["FI"].append(
                    {key : np.array(log_dict["FI"][key]).mean()  for key in log_dict["FI"].keys()},
                    ignore_index=True
                )
                elif key == "PRECISION@N":
                    for edge_size in log_dict["test"][key].keys():
                        df_metric[key] = df_metric[key].append(
                            {
                            **{
                                "experiment_No" : repeat_n-1,
                                "edge_size" : edge_size
                            },
                            **{_threshold : pd.Series(_precisions).mean() for _threshold,_precisions in log_dict["test"][key][edge_size].items()}
                            },
                            ignore_index=True
                        )
                else:
                    df_metric[key] = df_metric[key].append(
                        {_key : np.array(log_dict["test"][key][_key]).mean() for _key in log_dict["test"][key].keys()},
                        ignore_index=True
                    )
            # 输出当前实验的评价值
            print(f"\n{'Train':5s} mean_AUROC:{np.array(log_dict['train']['AUROC']).mean():.3f},mean_AUPR:{np.array(log_dict['train']['AUPR']).mean():.3f},mean_PRECISION:{np.array(log_dict['train']['PRECISION']).mean():.3f}")
            print(f"{'Test':5s} mean_AUROC:{np.array(log_dict['test']['AUROC']['all']).mean():.3f},mean_AUPR:{np.array(log_dict['test']['AUPR']['all']).mean():.3f},mean_PRECISION:{np.array(log_dict['test']['PRECISION']['all']).mean():.3f}")
                    # 记录每一次实验的特征重要性    
        
        # 调整precision@n的索引，第一级索引为edge_size，第二级索引为experiment_No
        df_metric["PRECISION@N"] = df_metric["PRECISION@N"].set_index(["edge_size","experiment_No"]).sort_index()
        # 将记录的dataFrame持久化到指定文件夹中,目录格式为 save_dir/dataset_name/now_time/dataFrame    
        print(df_prediction)

        if is_save:
            saveLog(
                df_prediction,
                df_metric,
                os.path.join(save_dir,dataset_name),
                model,
                model_params,
                timestamp
                )