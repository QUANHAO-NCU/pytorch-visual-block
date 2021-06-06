import sys
import argparse

from utils import *
from parameters import *
from Config import *
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


########################################################################################
# TODO 取消所有参数设置，使用Config配置文件
########################################################################################


### Parse the argument
dataset = mainArgs.dataset
model_type = mainArgs.model_type
class_number = mainArgs.class_number

### import parameters for respective datasets
# TODO 使用字体库
hyper_para = FounderHyperParameters()

if not mainArgs.default:
    hyper_para.lr = mainArgs.lr
    hyper_para.gpu_flag = mainArgs.gpu_flag
    hyper_para.iterations = mainArgs.iterations
    hyper_para.method = mainArgs.method
    hyper_para.verbose = mainArgs.verbose
    hyper_para.sigma = float(mainArgs.sigma)
    hyper_para.pre_trained_flag = mainArgs.pre_trained_flag
    hyper_para.N = mainArgs.N
    hyper_para.gamma = float(mainArgs.gamma)
    hyper_para.model_type = model_type
    hyper_para.classifier_type = mainArgs.classifier_type
    hyper_para.batch_size = mainArgs.batch_size

auc = 0.0

if mainArgs.model_mode == 'train':
    auc = choose_method(dataset, model_type, class_number, hyper_para)
elif mainArgs.model_mode == 'eval':
    print('work under progress')
    # auc = choose_models(dataset, model_type, class_number, hyper_para)

print(auc)
