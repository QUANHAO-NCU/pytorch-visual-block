class args(object):
    # the path of your dataset
    # 数据集的位置
    dataset = ''
    #
    model_type = 'OC_CNN'
    #
    class_number = None
    #######################################################
    # 使用默认参数?
    default = True
    # 学习率
    lr = 1e-4
    # 模型方法
    method = 'OC_CNN'
    # 模型模式 ['train','eval']
    model_mode = 'train'
    # 使用gpu?
    gpu_flag = True
    # epochs 迭代轮数
    iterations = 100

    N = None
    gamma = None
    batch_size = None

    classifier_type = None
    pre_trained_flag = True
    sigma = None
    verbose = None








