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
    downSampleForNE,
    evalution,
    evalutionAsBase,
    saveLog
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
    model_str , repeat_num , is_interator , model_params , dataset_names , is_save = \
        res_dict["model_str"] , res_dict["repeat_num"] , res_dict["is_interator"] , res_dict["model_params"] , res_dict["dataset_names"] , res_dict["is_save"]
        

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
        df_auroc = pd.DataFrame(columns=columns_str)
        df_aupr = pd.DataFrame(columns=columns_str)
        df_precision = pd.DataFrame(columns=columns_str)

        # 记录每一次实验的特征重要性
        if is_interator:
            feature_strs = []
            for  feature_class,feature_params in zip(model_params["feature_classes"],model_params["feature_params"]):
                feature_strs.append(
                    "-".join([feature_class] + list(map(str,feature_params.values())))
                )
                    
            df_fi = pd.DataFrame(columns=feature_strs)
        else:
            df_fi = {}
        # 第二重循环：重复做repeat_num次实验
        for repeat_n in range(1,repeat_num+1):
            print(f"\n\n\n{'dataset_name':<15s}:{dataset_name}\n{'model_str':<15s}:{model_str}\n{'repeat_n':<15s}:{repeat_n}/{repeat_num}\n")
            # 构造负样本（依照初始的超边基数分布）,形式为负边矩阵
            NE_matrix = downSampleForNE(PO_matrix,PO_matrix.shape[1])
            if suffix == "npz":
                NE_matrix = NE_matrix.tocsc()        
            kf = KFold(n_splits=kfold_num)

            # 下述两个列表用来记录五折交叉验证的指标,其中训练集不分基数记录，但测试集分基数记录
            train_log = {
                "AUROC" : [],
                "AUPR" : [],
                "PRECISION" : []
            }
            test_log = {
                "AUROC" : {x : [] for x in columns_str},
                "AUPR" : {x : [] for x in columns_str},
                "PRECISION" : {x : [] for x in columns_str}
            }
            
            # 记录每一次实验的特征重要性
            if is_interator:
                fi_log = defaultdict(list)
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
                train_result = evalution(train_label,train_prediction)
                
                test_prediction = model.test(test_sample)
                test_result = evalution(test_label,test_prediction)             
                test_result_b = evalutionAsBase(test_sample,test_label,test_prediction,JE_start_point.get(dataset_name,11))
                
                end_time = time.time()

                print_str = f"\nKflod[{kfold_counter+1}/{kfold_num}] in {end_time-start_time:.1f}s:\n" + \
                            f"    {'train':5s} AUROC:{train_result.get('AUROC',None):.3f},AUPR:{train_result.get('AUPR',None):.3f},PRECISION:{train_result.get('PRECISION',None):.3f}\n" + \
                            f"    {'test':5s} AUROC:{test_result.get('AUROC',None):.3f},AUPR:{test_result.get('AUPR',None):.3f},PRECISION:{test_result.get('PRECISION',None):.3f}"
                            
                print(print_str)
                
                for key in test_result_b.keys():
                    if test_result_b[key]:
                        test_log["AUROC"][key].append(test_result_b[key]["AUROC"])
                        test_log["AUPR"][key].append(test_result_b[key]["AUPR"])
                        test_log["PRECISION"][key].append(test_result_b[key]["PRECISION"])

                for key in train_result.keys():
                    train_log[key].append(train_result.get(key,None))
                    test_log[key]["all"].append(test_result.get(key,None))

                # 获得特征重要性
                if is_interator:
                    for key,value in model.getFeatureImportance().items():  #这里需要修改
                        fi_log[key].append(value)


            # 记录每一次实验的评价值
            df_auroc = df_auroc.append(
                {key : np.array(test_log["AUROC"][key]).mean()   for key in test_log["AUROC"].keys()},
                ignore_index=True
                )
            df_aupr = df_aupr.append(
                {key : np.array(test_log["AUPR"][key]).mean()   for key in test_log["AUPR"].keys()},
                ignore_index=True
                )
            df_precision = df_precision.append(
                {key : np.array(test_log["PRECISION"][key]).mean()   for key in test_log["PRECISION"].keys()},
                ignore_index=True
            )
            # 记录每一次实验的特征重要性
            
            if is_interator:
                df_fi = df_fi.append(
                    {key : np.array(fi_log[key]).mean()  for key in fi_log.keys()},
                    ignore_index=True
                )

            # 输出当前实验的评价值
            print(f"\n{'Train':5s} mean_AUROC:{np.array(train_log['AUROC']).mean():.3f},mean_AUPR:{np.array(train_log['AUPR']).mean():.3f},mean_PRECISION:{np.array(train_log['PRECISION']).mean():.3f}")
            print(f"{'Test':5s} mean_AUROC:{np.array(test_log['AUROC']['all']).mean():.3f},mean_AUPR:{np.array(test_log['AUPR']['all']).mean():.3f},mean_PRECISION:{np.array(test_log['PRECISION']['all']).mean():.3f}")
        
        
        
        
        # 将记录的dataFrame持久化到指定文件夹中,目录格式为 save_dir/dataset_name/now_time/dataFrame    
        if is_save:
            saveLog(
                {
                    "df_auroc" : df_auroc,
                    "df_aupr"  : df_aupr,
                    "df_precision" : df_precision,
                    "df_fi" : df_fi
                },
                os.path.join(save_dir,dataset_name),
                model,
                model_params,
                timestamp
                )