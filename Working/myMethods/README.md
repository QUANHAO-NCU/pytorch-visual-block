程序使用pytorch框架
你可能需要安装的第三方库有：
pytorch ： 网络框架,版本1.8.1
visdom：用于可视化训练过程
sklearn：用于计算Accuracy，Precision，Recall，F1等参数
matplotlib：用于绘制分析图像

程序会自行联网下载数据集，所以无须设置数据集位置
可以换用CIFAR10数据集进行测试，注意Encoder和Decoder也要同步更换为同样版本的

运行流程：
step 1 启动visdom服务器，用于可视化训练结果
在控制台输入

```shell
python -m visdom.server
```

启动后会在控制台输出一个地址，用浏览器打开该地址
step 2 训练网络

```shell
python Train.py
```

