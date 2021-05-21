# 使用CoLab训练网络

CoLab相当于一个虚拟机，你可以在上面使用Google提供的GPU训练你的神经网络。（我用的时候分配到了一块T4显卡，显存有16G，足以应付大部分情况）

#### Step 1 配置环境

使用!执行系统命令，操作类似于虚拟机。

安装pytorch，matplotlib，tqdm，visdom等第三方库。

pytorch 可能安装有默认版本的，可以不装。

matplotlib一般默认安装，也可以不装。

tqdm用于监视程序运行情况，建议安装。

visdom用于可视化训练情况，可装可不装，安装相对麻烦。

numpy一般内置了，无须安装。

```shell
!pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
!pip3 install matplotlib
!pip3 install visdom
!pip3 install tqdm
!pip3 install numpy
```

#### Step2 挂载云盘

![image-20210521203542728](使用Colab训练网络_images\image-20210521203542728.png)

这样你的代码和数据集可以上传到Google硬盘，不必每次启动CoLab时都要调试。

#### Step 3  添加程序工作路径

从目录下打开代码，复制，新建代码，粘贴上去。

![image-20210521204305788](使用Colab训练网络_images\image-20210521204305788.png)

Colab的代码的工作路径与硬盘的路径不一样的，使用这两行代码添加工作路径，使之能够正常导入我们自定义的包。

#### Step 4 指定参数

指定网络参数，开始训练

![image-20210521205100896](使用Colab训练网络_images\image-20210521205100896.png)

#### Attention 1 不要关闭标签页

CoLab不是完全能脱机运行的，直接关闭标签页，或者长期不查看这个标签页，都会导致虚拟机关闭，训练中断，所以使用CoLab训练的时候，可以时不时点一点这个页面。

使用连接到托管程序或许有用。连接到托管程序后，不需要去点击网页了，但是标签页最好不关。（似乎关闭了也没有影响）

#### Attention 2 时间限制

CoLab最长可以连续使用12小时，12小时之后会清除虚拟机，所以训练时记得保存权值文件，