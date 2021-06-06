class mainArgs(object):
    # the path of your dataset
    # 数据集的位置
    dataset = r'D:\Code\machineLearning\pyTorch\Dataset\1001 Abnormal Image Dataset'
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
    model_mode = 'vggface'
    # 使用gpu?
    gpu_flag = True
    # epochs 迭代轮数
    iterations = 100

    N = 1.0
    D = 2
    gamma = None
    batch_size = None

    classifier_type = None
    pre_trained_flag = True
    sigma = None
    verbose = None
    sigma1 = None


class trainArgs(object):
    # 特征提取器
    extractor = 'vgg16'
    extractor_pretrain = True
    # 数据集路径
    datasetPath = r'D:\Code\machineLearning\pyTorch\Dataset\1001 Abnormal Image Dataset'
    # Dataloader 加载数据时启用的线程数
    num_workers = 4
    # 训练批次大小
    batch_size = 32
    # 启用显卡
    cuda = True
    # 学习率
    lr = 1e-4
    # 优化器SGD动量
    momentum = 0.9
    # 惩罚参数
    weight_decay = 0
    # 断点续训
    resume = False
    # 特征提取器提取到的特征，vgg，AlexNet为4096，resnet为512
    D = 4096
    # 训练起始轮数
    start_epoch = 0
    # 训练终止轮数
    epochs = 100
    # 指定正类
    positiveClass = 'Car_Abnormal'
    #######################################################




    # 权重文件保存路径
    save_path = ''

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

    classifier_type = None
    pre_trained_flag = True
    sigma = None
    verbose = None
