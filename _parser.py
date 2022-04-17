import argparse
from model import *
from typing import Dict
from model.indicator.Base import Indicator
from model.integrator.Base import Integrator
from model.intensifier.Base import Intensifier
save_dir = r"./output"
dataset_dir = r"./dataset"
dataset_names = ["chuancai","yuecai","iAB_RBC_283","iJO1366","arXiv_cond-mat","email-Eu","cat-edge-music-blues-reviews"]
JE_start_point = {
        "chuancai"        : 4,
        "yuecai"          : 4,
        "iAB_RBC_283"     : 6,
        "iJO1366"         : 6,
        "arXiv_cond-mat"  : 5
    }

preprocess_log = {
    "chuancai" : "m",
    "yuecai"  : "m",
    "iAB_RBC_283" : None,
    "iJO1366" : None,
    "arXiv_cond-mat" : None,
    "email-Eu" : "m",
    "cat-edge-music-blues-reviews" : None
}
kfold_num = 5
random_seed = 2


parser = argparse.ArgumentParser(description="该程序目的是集合不同指标作为特征以进行链路预测")
parser.add_argument("-m","--method",type=str,choices=["NNAA","NEAA","ENAA","EEAA","CLAA","CSAA","RA","LR","RF","LGBM","CN","CE","CO","CF","WCF","JC","COS","WK2","WK3","DE","HKatz","SKatz","HWalk","HEffi","SHEffi","TRW","RRW","LTRW","SLTRW","SRA","StRA","SuRA","PS","SPS","SC","SSC"],help="指定一个用来链路预测的方法",default="NNAA")
parser.add_argument("-i","--indicators",type=str,help="指定方法为集成或增强方法时该项起作用，指定特征的类型",default="NNAA,NEAA,ENAA,EEAA,HWalk")
parser.add_argument("-d","--datasets",type=str,help="指定对应的数据集",default="chuancai,yuecai,cat-edge-music-blues-reviews,email-Eu,iAB_RBC_283,iJO1366,arXiv_cond-mat,CoreComplex")
parser.add_argument("-r","--repeat",type=int,default=10,help="实验的重复次数")  
parser.add_argument("-ns","--no_save",action="store_true",help="是否保存实验结果")  
parser.add_argument("-wid","--width",type=str,default="1,2,3",help="指定方法为或指定指标中包含HWalk或HEffi时生效，用来指定HWalk的宽度")
parser.add_argument("-len","--length",type=str,default="1,2,3",help="指定方法为或指定指标中包含HWalk时生效，用来指定HWalk的长度")
parser.add_argument("-bn","--base_number",type=str,default="0.1,0.2",help="指定方法为或指定指标中包含SHEffi且衰减系数类型为exp时生效，用来指定衰减exp的底数")
parser.add_argument("-rp","--restart_prob",type=str,default = "0.2",help="指定方法为带重启的随机游走类指标时生效，用来指定随机游走的重启概率")
parser.add_argument("-s","--steps",type=str,default = "5",help="指定方法为局部随机游走指标（LTRW）或简单资源分配（SRA）或概率传播（PS）时生效，用来指定局部随机游走的游走步数或简单资源分配、概率传播的步数")
parser.add_argument("-a","--alpha",type=str,default="0.2",help="指定方法为叠加RA（SuRA）时生效，用来指定叠加资源分配的衰减系数")
parser.add_argument("-pt","--precision_threshold",type=int,default=50,help="计算precision@n时的阈值数量")
cmd_args = parser.parse_args()


def processingCommandParam(cmd_args:argparse.Namespace)->Dict:
    model_str = cmd_args.method
    repeat_num = cmd_args.repeat
    precision_threshold = cmd_args.precision_threshold
    is_save = not (cmd_args.no_save)
    # 此部分用于得到HWalk或HEffi的指定参数
    width_list = cmd_args.width.split(",")[::-1]
    length_list = cmd_args.length.split(",")[::-1]
    re_prob_list = cmd_args.restart_prob.split(",")[::-1]
    base_number_list = cmd_args.base_number.split(",")[::-1]
    steps_list = cmd_args.steps.split(",")[::-1]
    alpha_list = cmd_args.alpha.split(",")[::-1]
    if str_to_object[model_str] in Integrator.__subclasses__():
        is_interator = True
        model_params = { "feature_classes" : cmd_args.indicators.split(",")}    
        model_params["feature_params"] = []
        for  feature_class in model_params["feature_classes"]:
            if feature_class == "HWalk":
                model_params["feature_params"].append({
                    "width": int(width_list.pop()),
                    "length"    : int(length_list.pop())
                })
            elif feature_class == "HEffi":
                model_params["feature_params"].append({
                    "width"    : int(width_list.pop())
                })
            elif feature_class in ["LTRW","SLTRW"]:
                model_params["feature_params"].append({
                    "restart_prob" : float(re_prob_list.pop()),
                    "steps" : int(steps_list.pop())
                })
            elif feature_class == "SHEffi":
                model_params["feature_params"].append(
                    {
                        "width": int(width_list.pop()),
                        "base_number"   : float(base_number_list.pop())
                    }
                )
            elif feature_class in ["StRA","TRW","RRW"]:
                model_params["feature_params"].append(
                    {
                        "restart_prob" : float(re_prob_list.pop())
                    }
                )
            elif feature_class in ["SRA","PS"]:
                model_params["feature_params"].append(
                    {
                        "steps": int(steps_list.pop())
                    }
                )
            elif feature_class == "SuRA":
                model_params["feature_params"].append(
                    {
                        "alpha" : float(alpha_list.pop())
                    }
                )
            elif feature_class in ["SC","SSC"]:
                model_params["feature_params"].append(
                    {
                        "steps" : int(steps_list.pop()),
                        "alpha" : float(alpha_list.pop())
                    }
                )
            else:
                model_params["feature_params"].append({})
    elif str_to_object[model_str] in Intensifier.__subclasses__():
        is_interator = False
        model_params = {"raw_indicator_abb" : cmd_args.indicators.split(",")[0]}
    else:
        is_interator = False
        if model_str == "HWalk":
            model_params = {
                "width" : int(width_list.pop()),
                "length"    : int(length_list.pop())
            }
        elif model_str == "HEffi":
            model_params = {
                "width"    : int(width_list.pop())
            }
        elif model_str == "SHEffi":
            model_params = {
                        "width" : int(width_list.pop()) ,
                        "base_number" : float(base_number_list.pop())
                    }
        elif model_str in ["LTRW","SLTRW"]:
            model_params = {
                "restart_prob" : float(re_prob_list.pop()),
                "steps" : int(steps_list.pop()),
            }
        elif model_str in ["SRA","SPS"]:
            model_params = {
                    "steps": int(steps_list.pop())
                }
        elif model_str in ["StRA","TRW","RRW"]:
            model_params = {
                    "restart_prob" : float(re_prob_list.pop())
                }
        elif model_str == "SuRA":
            model_params = {
                    "alpha" : float(alpha_list.pop())
                }
        elif model_str in ["SC","SSC"]:
            model_params = {
                    "steps" : int(steps_list.pop()),
                    "alpha" : float(alpha_list.pop())
            }
        else:
            model_params = {}
    dataset_names = cmd_args.datasets.split(",")
    return {
        "model_str" : model_str,
        "repeat_num"    : repeat_num,
        "is_interator"  : is_interator,
        "model_params"  : model_params,
        "dataset_names" : dataset_names,
        "is_save"       : is_save,
        "precision_threshold" : precision_threshold
    }