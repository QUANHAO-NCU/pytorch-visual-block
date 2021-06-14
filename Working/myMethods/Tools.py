import copy
from torch import nn
import PIL
import numpy as np
import torch
from PIL.Image import Image
from scipy.signal import convolve2d
import os
import random
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import roc_auc_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def addNoise(images, std, device):
    noise = np.random.normal(0, std, size=images.shape)
    noise = torch.from_numpy(noise).float().to(device)
    image_noise = (images + noise).clamp(0.0, 1.0)
    images = images.to(device)
    image_noise = image_noise.to(device)
    return images, image_noise


def addGaussianNoise2Image(images, labels, std=0.1, shuffle=True, device=None):
    """
    给图片添加高斯噪声
    原图片的标签为1:正样本
    添加了噪声的图片的标签为0:负样本
    """
    images = images.float().to(device)
    labels = labels.float().to(device)
    batch_size = labels.size(0)
    labels = torch.unsqueeze(labels, dim=0).view(batch_size, 1)
    noiseLabel = torch.zeros((batch_size, 1)).to(device)

    noise = torch.zeros(images.shape).to(device)
    noise = noise + (std ** 2) * torch.rand_like(images)
    noise = torch.add(noise, images)

    data = torch.cat((images, noise), dim=0)
    labels = torch.cat((labels, noiseLabel), dim=0)
    if shuffle:
        tmp = []
        for index, tensor in enumerate(data):
            tmp.append([tensor, labels[index]])
        random.shuffle(tmp)
        data = [torch.unsqueeze(i[0], dim=0) for i in tmp]
        labels = [i[1] for i in tmp]
        labels = torch.cat(labels, dim=0)
        data = torch.cat(data, dim=0)
    if labels.dim() >= 2:
        labels = torch.squeeze(labels, dim=1)
    labels = labels.long()
    data = data.float()
    return data, labels


def PreRecAccAuc(target, scores, true_threshold):
    """
    二分类评判指标
    precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    Accuracy = (TP + TN) / len(target)
    TPR(True positive rate) = TP / (TP + FN)
    FPR(False positive rate) = FP / (FP + TN)
    AURoc(Area Under Receiver operating characteristic Curve) = area of ROC curve
    假定正样本标签为1
       负样本标签为0
    """

    predict_label = torch.ge(scores, true_threshold).float()
    Accuracy = torch.eq(predict_label, target).float().sum().item() / len(target)
    target = target.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    predict_label = predict_label.cpu().detach().numpy()
    Precision = metrics.precision_score(target, predict_label, pos_label=1, zero_division=0)
    Recall = metrics.recall_score(target, predict_label, pos_label=1, zero_division=0)
    FPR, TPR, threshold = metrics.roc_curve(target, scores, pos_label=1)
    AUC = metrics.roc_auc_score(target, scores)
    return Precision, Recall, Accuracy, AUC


def addGaussianNoise2Vector(vectors, labels, mean=0, std=0.1, shuffle=False, device=None):
    """
    给特征向量添加高斯噪声
    原始特征向量对应的标签是1
    使用噪声生成的特征向量标签为0
    以此训练分类器网络
    """
    # vectors --> [batch_size,512]
    # labels --> [batch_size]
    # 一个正常输入对应一个噪声
    # 假定正样本遵循的分布一定不是高斯分布
    batch_size = labels.size(0)
    # label 先扩展维度，便于拼接[32]--> [1,32]-->[32,1]
    labels = torch.unsqueeze(labels, dim=0).view(batch_size, 1)
    noiseLabel = torch.zeros((batch_size, 1)).to(device)
    noise = np.random.normal(mean, std, vectors.shape)
    noise = torch.from_numpy(noise).to(device)
    # noise = torch.add(noise, vectors)
    noise = torch.sigmoid(noise)
    data = torch.cat((vectors, noise), dim=0)
    labels = torch.cat((labels, noiseLabel), dim=0)
    # 将输入和噪声打乱
    if shuffle:
        tmp = []
        for index, tensor in enumerate(data):
            # tensor [6272]
            tmp.append([tensor, labels[index]])
        random.shuffle(tmp)
        # data item [1,512]
        data = [torch.unsqueeze(i[0], dim=0) for i in tmp]
        # labels item [1]
        labels = [i[1] for i in tmp]
        labels = torch.cat(labels, dim=0)
        data = torch.cat(data, dim=0)
    if labels.dim() >= 2:
        labels = torch.squeeze(labels, dim=1)
    labels = labels.long()
    data = data.float()
    return data, labels


def SSIM(Images1, Images2):
    """
    计算两组图像的SSIM （结构性误差）
    https://zh.wikipedia.org/wiki/%E7%B5%90%E6%A7%8B%E7%9B%B8%E4%BC%BC%E6%80%A7
    https://blog.csdn.net/weixin_42096901/article/details/90172534
    """
    # SSIM取值范围[-1,1]，越接近于1表示越接近
    # image1,image2 --> [batch_size,3,H,W]
    # 计算图像的R、G、B三个通道的均值和标准差
    # 计算两幅图像在R、G、B三个通道上的SSIM
    # 取这三个通道上的SSIM值的平均值作为两幅图像的平均值
    # 默认图像为RGB模式
    assert len(Images1) == len(Images2)
    average_ssim = 0
    batch_size = len(Images1)
    images1 = torch.add(Images1, 0)
    images2 = torch.add(Images2, 0)

    def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def filter2(x, kernel, mode='same'):
        return convolve2d(x, np.rot90(kernel, 2), mode=mode)

    def compute_ssim(in1, in2, k1=0.01, k2=0.03, win_size=11, L=255):
        if not in1.shape == in2.shape:
            raise ValueError("Input Imagees must have the same dimensions")
        if len(in1.shape) > 2:
            raise ValueError("Please input the images with 1 channel")

        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
        window = window / np.sum(np.sum(window))
        if in1.dtype == np.uint8:
            in1 = np.double(in1)
        if in2.dtype == np.uint8:
            in2 = np.double(in2)
        # SSIM计算的图片像素范围原来是0-255，输入图片如果是0-1之间的先转换为0-255
        image1 = in1
        image2 = in2
        if in1.max() <= 1 and in2.max() <= 1:
            image1 *= 255
            image2 *= 255

        miu1 = filter2(image1, window, 'valid')
        miu2 = filter2(image2, window, 'valid')
        miu1_sq = miu1 * miu1
        miu2_sq = miu2 * miu2
        mean12 = miu1 * miu2
        sigma1_sq = filter2(image1 * image1, window, 'valid') - miu1_sq
        sigma2_sq = filter2(image2 * image2, window, 'valid') - miu2_sq
        sigma12 = filter2(image1 * image2, window, 'valid') - mean12

        ssim = ((2 * mean12 + C1) * (2 * sigma12 + C2)) / ((miu1_sq + miu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return np.mean(np.mean(ssim))

    for i in range(batch_size):
        im1 = images1[i]
        im2 = images2[i]
        im1 = im1.cpu().detach().numpy()
        im2 = im2.cpu().detach().numpy()
        R_ssim = compute_ssim(im1[0], im2[0])
        G_ssim = compute_ssim(im1[1], im2[1])
        B_ssim = compute_ssim(im1[2], im2[2])
        ssim = (R_ssim + G_ssim + B_ssim) / 3
        average_ssim += ssim
    average_ssim /= batch_size
    return average_ssim


def frozenModel(model: nn.Module):
    """"
    冻结模型
    """
    for param in model.parameters():
        param.requires_grad = False


def myLoss(scores, true_threshold, false_threshold, true_labels):
    predict_true = torch.ge(scores, true_threshold).float()
    predict_false = torch.le(scores, false_threshold).float()
    TP = torch.ge(torch.add(predict_true, true_labels), 2).float()
    TN = torch.ge(torch.sub(predict_false, true_labels), 1).float()
    FP = torch.ge(torch.sub(predict_true, true_labels), 1).float()
    FN = torch.ge(torch.add(predict_false, true_labels), 2).float()
    TP_scores = torch.mul(TP, scores)
    TN_scores = torch.mul(TN, scores)
    FP_scores = torch.mul(FP, scores)
    FN_scores = torch.mul(FN, scores)
    loss = torch.add((torch.sum(TP) - torch.sum(TP_scores)), (torch.sum(FN) - torch.sum(FN_scores)))
    loss = torch.add(loss, torch.add(torch.sum(TN_scores), torch.sum(FP_scores)))
    mean_TP = torch.div(torch.sum(TP_scores), torch.sum(TP))
    mean_TN = torch.div(torch.sum(TN_scores), torch.sum(TN))
    if torch.isnan(mean_TP):
        mean_TP = true_threshold
    else:
        mean_TP = mean_TP.item()
    if torch.isnan(mean_TN):
        mean_TN = false_threshold
    else:
        mean_TN = mean_TN.item()
    return loss, mean_TP, mean_TN


def ConvHWKSP(HW, k, s, p):
    Hin = HW[0]
    Win = HW[1]
    Hout = (Hin - k + 2 * p) // s + 1
    Wout = (Win - k + 2 * p) // s + 1
    return [Hout, Wout]


def ConvTransposeHWKSP(HW, k, s, p):
    Hin = HW[0]
    Win = HW[1]
    Hout = (Hin - 1) * s - 2 * p + k
    Wout = (Win - 1) * s - 2 * p + k
    return [Hout, Wout]


def ConvHW2newHW(HWin: list, HWout: list):
    """
    计算卷积时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - k + 2 * p) // s + 1
                tmpW = (Win - k + 2 * p) // s + 1
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


def ConvTransposeHW2newHW(HWin: list, HWout: list):
    """
    计算转置卷积时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - 1) * s - 2 * p + k
                tmpW = (Win - 1) * s - 2 * p + k
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


def PoolHW2newHW(HWin: list, HWout: list):
    """
    计算池化时从旧HW转换到新HW时的可以使用的kernel_size,stride,padding参数
    """
    result = []
    Hin = HWin[0]
    Win = HWin[1]
    Hout = HWout[0]
    Wout = HWout[1]
    for k in range(1, 15):
        for s in range(1, 10):
            for p in range(0, 10):
                tmpH = (Hin - k + 2 * p) // s + 1
                tmpW = (Win - k + 2 * p) // s + 1
                if tmpH == Hout and tmpW == Wout:
                    result.append([k, s, p])
    return result


if __name__ == '__main__':
    # scores = torch.from_numpy(np.asarray([0.87, 0.24, 0.67, 0.73, 0.17, 0.23])).float()
    # true_labels = torch.from_numpy(np.asarray([0, 0, 1, 1, 0, 1])).float()
    # true_threshold = 0.6
    # false_threshold = 0.3
    # loss = myLoss(scores, true_threshold, false_threshold, true_labels)
    # out = PreRecAccAuc(true_labels, scores, true_threshold)
    images = torch.randn((128, 3, 32, 32))
    labels = torch.ones(128)
    data, labels = addGaussianNoise2Image(images, labels, 0.1, shuffle=True, device=torch.device('cuda:0'))
    pass
