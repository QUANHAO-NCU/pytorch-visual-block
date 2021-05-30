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
dataset = args.dataset
model_type = args.model_type
class_number = args.class_number

### import parameters for respective datasets
# TODO 使用字体库
hyper_para = FounderHyperParameters()

if not args.default:
    hyper_para.lr = args.lr
    hyper_para.gpu_flag = args.gpu_flag
    hyper_para.iterations = args.iterations
    hyper_para.method = args.method
    hyper_para.verbose = args.verbose
    hyper_para.sigma = float(args.sigma)
    hyper_para.pre_trained_flag = args.pre_trained_flag
    hyper_para.N = args.N
    hyper_para.gamma = float(args.gamma)
    hyper_para.model_type = model_type
    hyper_para.classifier_type = args.classifier_type
    hyper_para.batch_size = args.batch_size

auc = 0.0

if args.model_mode == 'train':
    auc = choose_method(dataset, model_type, class_number, hyper_para)
elif args.model_mode == 'eval':
    print('work under progress')
    # auc = choose_models(dataset, model_type, class_number, hyper_para)

print(auc)
