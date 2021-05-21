import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from visdom import Visdom

"""
@author: QH
@CreateTime:2021年3月24日14:36:11
"""

'''
参数说明
  全局参数：
    max_row：(必须设置) 最大行数，可以设置为要显示的最大通道数；如果要显示网络中某一层的全部通道(Channel)，将这个值设置为网络中最大的通道数
    max_column:(必须设置)最大列数，数值为要显示的网络层数，如果不确定，可以将这个值设置的大一点
    w：              显示的图的宽，单位为像素
    h：              显示的图的高，单位为像素
    margin_right：   图片的右边距
    margin_top:      图片的上边距
        事实上这四个值和后面的dpi相关，而dpi与运行的设备的分辨率有关，因此显示在浏览器上的图片的宽高并不一定与设置的值相同
        大部分时候都不相同，在1920*1080分辨率，dpi 40下，真实图片大小约为设置值的2.4倍
    dpi: 每英寸上的像素数，与设备有关
    调整w，h，dpi都可以调整图像的大小
  局部参数：
    mode: （必须设置）显示模式 目前实现了feature_map（特征图）、source_image（原图）；kernel_weights在开发中
    layer：（必须设置）当前层的名字
    channel_num: 当前层显示的通道数 
        'all' 输出当前层的全部通道，需要小于设置的 max_row,否则最多显示max_row个
        1,2,3,...et;
    cmap： 设置图的显示模式
        如果mode是source_img模式，则此项无效
        如果是feature_map模式，不设置，则将灰度图映射到蓝、绿颜色空间中显示
        gray 灰度图
'''


class visual_block(nn.Module):
    def __init__(self, max_row=10, max_column=10, w=60, h=60, margin_right=2, margin_top=2, dpi=30):
        super(visual_block, self).__init__()
        self.axes_pool = {}
        self.max_column = max_column
        self.max_row = max_row
        self.dpi = dpi
        self.layer_count = 0
        self.layer_pool = {}
        self.image_figure_pool = {}
        self.figure_width = self.max_column * (w + margin_right) * 1.0 / dpi
        self.figure_height = self.max_row * (h + margin_top) * 1.0 / dpi
        self.image_w = w * 1.0 / (self.figure_width * dpi)
        self.image_h = h * 1.0 / (self.figure_height * dpi)
        self.width_bias = margin_right / (self.figure_width * dpi)
        self.height_bias = margin_top / (self.figure_height * dpi)
        self.viz = Visdom()

    def update_graph(self, figure, data, params):
        data = data.cpu()
        # 解析参数
        options = {'mode': 'feature_map', 'layer': 'conv1', 'channel_num': 'all',
                   'cmap': None, 'image': ''}
        for i in params:
            if i in options:
                options[i] = params[i]

        if options['channel_num'] == 'all':
            channel_num = len(data)
        else:
            channel_num = options['channel_num']

        index = options['image'] + '_' + options['layer']

        if index not in self.axes_pool:
            if options['layer'] not in self.layer_pool:
                self.layer_count += 1
                self.layer_pool[options['layer']] = self.layer_count
            cli = current_layer_index = self.layer_pool[options['layer']]
            self.axes_pool[index] = []

            for i in range(len(data)):
                width = 1.0 / self.max_column - self.width_bias
                height = 1.0 / self.max_row - self.height_bias
                left = (cli - 1) * 1.0 / self.max_column
                bottom = (self.max_row - i - 1) * 1.0 / self.max_row
                react = [left, bottom, width, height]
                axi = figure.add_axes(react)
                axi.axis('off')
                self.axes_pool[index].append(axi)
        axes = self.axes_pool[index]

        if options['mode'] == 'source_image':
            data = self.denormalize(data)
            img = data.detach().numpy()
            img = np.transpose(img, [1, 2, 0])
            axes[0].imshow(img)
        else:
            data = data.detach().numpy()
            if channel_num > self.max_row:
                channel_num = self.max_row
            for channel in range(channel_num):
                axes[channel].imshow(data[channel], cmap=options['cmap'])
        return figure

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def forward(self, x):

        """
        可视化
        Options:
            mode:   'feature_map','kernel_weights','source_image'
                指定输入的tensor是属于特征图、卷积核，或者是原图
            layer: 'conv1',''
                当前层名称,必填。用于索引 ax 子图
            channel: 'all',1,2,...
                指定要显示的特征图或者卷积核参数数量
        """
        params = x[1]
        x = x[0]
        show_data = x
        for image in range(len(show_data)):
            if 'image%d' % image not in self.image_figure_pool:
                fig = plt.figure(figsize=(self.figure_width, self.figure_height), dpi=self.dpi)
                fig.tight_layout()
                self.image_figure_pool['image%d' % image] = fig
            params['image'] = 'image%d' % image
            figure = self.image_figure_pool['image%d' % image]
            self.update_graph(figure, show_data[image], params)
            if 'end' in params and params['end']:
                self.viz.matplot(plot=self.image_figure_pool['image%d' % image], win='image%d' % image,
                                 env='image%d' % image)
        return x
