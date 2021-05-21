# pytorch-visual-block
继承自pytorch nn.Module 的特征图可视化插件。Visual plug-ins inherited from pytorch nn.Module.

## 效果

![image-20210324150201398](.README_asserts\image-20210324150201398.png)

Batchsize为2，一次训练两张图片，切换不同的Environment来查看不同的图片情况

![image-20210324150227968](.README_asserts\image-20210324150227968.png)

可以在训练过程中查看特征图的情况

## 参数说明

####   全局参数：    

​	max_row：(必须设置) 最大行数，可以设置为要显示的最大通道数；如果要显示网络中某一层的全部通道(Channel)，将这个值设置为网络中最大的通道数
​    max_column:(必须设置)最大列数，数值为要显示的网络层数，如果不确定，可以将这个值设置的大一点
​    w：              显示的图的宽，单位为像素
​    h：              显示的图的高，单位为像素
​    margin_right：   图片的右边距
​    margin_top:      图片的上边距
​        事实上这四个值和后面的dpi相关，而dpi与运行的设备的分辨率有关，因此显示在浏览器上的图片的宽高并不一定与设置的值相同
​        大部分时候都不相同，在1920*1080分辨率，dpi 40下，真实图片大小约为设置值的2.4倍
​    dpi: 每英寸上的像素数，与设备有关
​    调整w，h，dpi都可以调整图像的大小
​    device: 运行的设备，cpu / cuda:0 （GPU）

####   局部参数：

​    mode: （必须设置）显示模式 目前实现了feature_map（特征图）、source_image（原图）；kernel_weights在开发中
​    layer：（必须设置）当前层的名字
​    channel_num: 当前层显示的通道数 
​        'all' 输出当前层的全部通道，需要小于设置的 max_row,否则最多显示max_row个
​        1,2,3,...et;
​    cmap： 设置图的显示模式
​        如果mode是source_img模式，则此项无效
​        如果是feature_map模式，不设置，则将灰度图映射到蓝、绿颜色空间中显示
​        gray 灰度图

## 使用样例

```python
#设置全局参数
self.visual_block = visual_block(max_row=10, max_column=10)
```

![设置全局参数](.README_asserts\image-20210324145434085.png)![image-20210324145434132](.README_asserts\image-20210324145434132.png)

```python
#在forward函数中，显示原图片
self.visual_block([x, {'mode': 'source_image', 'layer': 'source_image'}])
# 输出卷积层1的特征图
self.visual_block([x, {'mode': 'feature_map', 'layer': 'conv_1', 'channel_num': 10}])
```

![在forward函数中使用](.README_asserts\image-20210324145833051.png)





