## Preface

下面是我对这篇 Reading Text in the Wild with Convolutional Neural Networks 文章, 对于前半部分：文字定位检测部分的复现大致流程。

用的数据集是 ICDAR 2011: http://robustreading.opendfki.de/trac/wiki/SceneText ，不少人都说 ICDAR 2011 数据集下载不了，我在这里上传一份我自己备份的：[ICDAR 2011](http://share.weiyun.com/af0cc4a85ba65b26312f192a0d14c084)

需要指出的是，一方面因为做个实验与示例，且数据集小，做的结果比较粗糙。希望大家包含一下，因为不少同学跟我私信要代码，我在这里贴出来。希望得到大神的建议，帮助完善。

整理后的过程文件都在 reading text in the wild 中。

edge_boxes_with_python 文件夹，存放 Edge Boxes 、Random Forest 的代码，还有一些中间保存的变量结果。

Bounding_Box_Reg 是存放最后回归的文件夹。训练数据的生成、网络的定义都在里面。

Output 文件夹存放中间输出的图像，即将 Bounding Boxes 画在原图上的结果。

**注意：**下面有些数学公式，Github 上不支持，您下载下来，用支持数学公式的 Markdown 编辑器打开即可。

## Edge Boxes

### Overview 

这部分的方法出自 Paper: [Edge boxes: Locating object proposals from edges](https://www.microsoft.com/en-us/research/publication/edge-boxes-locating-object-proposals-from-edges/)。

先计算 edge response，用 [Structured Edge detector](https://www.microsoft.com/en-us/research/publication/structured-forests-for-fast-edge-detection/) 检测算法。

同时，使用了 Paper: [Fast Edge Detection Using Structured Forests](https://arxiv.org/pdf/1406.5549.pdf) 介绍的 **single-scale** 变量来减少运行时间。

在得到 edge response 之后，我们进行一个 **非极大值抑制（Non-Maximal Suppression, NMS）**。

结果为 **稀疏的边缘图（sparse edge map）**，每个像素 $p$ 有一个边缘模 $m_p$，以及方向 $\Theta_p$。当像素地模 $m_p > 0.1$ 的时候，就定义为 **edges**。

还有个概念，所谓的 **contour** ，定义为一系列的 edges 所形成的连贯的边界线、曲线或者直线。

### Implement

这部分程序的实现，用 Python 调用了 Dollor 的 Matlab 源码：[Structured Edge Detection Toolbox](https://github.com/pdollar/edges) 

程序入口为：`edge_boxes.py` 。

该文件里面，`def  get_windows(image_fnames, cmd='edge_boxes_wrapper')` 为调用 Matlab 源码函数。

先粗的检测过之后，需要进行 非极大值抑制（Non-Maximum Suppression），非极大值抑制函数在 Python 文件：`nms.py` 中。

之后，将不进行非极大值抑制与进行极大值抑制过后的 Boxes ，通过 `pickle` 模块分别保存到 `boxes_edgeBoxes_without_NMS.pkl`、 `boxes_edgeBoxes_with_NMS.pkl` 文件中，以供后面使用。

同时，我将这些 Boxes 在图像上画出来，保存在 `./Output/edge_boxes_with_NMS`，以及`./edge_boxes_without_NMS` 中。

再之后，我写了一个名为 `calRecall.py`  的文件，用来计算 Recall，以及将 $IoU$ 小于 $0.5$ 的 Boxes 去除掉。将结果同样用 `pickle`  保存，生成文件名称为 `boxes_IOU_gt05_withoutNMS.pkl`，以及 `boxes_IOU_gt05_withNMS.pkl`。

同时，在该文件中，我将每一张图像，其检测出来的每一个 $IoU \gt 0.5$ 的 Bounding Boxes，对应的 Grounding Truth 一一对应保存到 `map_gtBox_BBoxes.pkl` 当中。因为在最后的 Bounding Boxes Regression 里面，生成数据需要这个。

将这部分处理后的 boxes 画在图上，见文件夹 `./Output/edge_boxes_IOU_gt05`。



## Aggregate Channel Features Detector

这部分是基于作者 BMVC 2009 年的 Paper: [Integral Channel Features](http://pages.ucsd.edu/~ztu/publication/dollarBMVC09ChnFtrs_0.pdf)，就是将输入的图像进行各种变换处理，得到不同特征的图像。再进行区域求和，再训练分类器。

这部分的代码，我尝试了作者提供的源码：http://pdollar.github.io/toolbox/，在作者发布的 Matlab 工具箱里。

但是，我跑了一个白天，没调通。。。-_-#

之后，我又试了文章里提到的 [BING](http://mmcheng.net/zh/bing/)，分别编译了两个版本的源码：

https://github.com/tfzhou/BINGObjectness

https://github.com/bittnt/Objectness

这个程序很快跑通了，但对于南开大学的这篇 BING 我没仔细看。而且研究了一会儿，没明白该怎么组织我这个数据才能跑。

考虑到进度，这部分就放弃了。我跳过这一步，进行下一步。



## Word Classification

### HOG 特征提取、Random Forest 分类

接着上面的结果，这一部分是对上面经过 Edge Boxes ＋Aggregate Channel Feature 后的进一步处理。目的是去除错误的 bounding box proposals。

#### 思路

文章中进一步过滤 bounding boxes 使用的是 HOG 特征＋Random Forest Classifier。阅读论文后，我的处理过程大致如下：

**第一个阶段：训练 Random Forest Classifier**

根据数据集给的每张图像文字区域的 ground truth（ txt 文件中给的是 $[X_{min}, Y_{min}, X_{max}, Y_{max}, score]$）。将这些区域提取出来，经过 Resize 的处理，提取出 HOG 特征。

同时，根据论文的方法。还需要加入负样本，我加入了 $300$ 张的负样本。对于这些负样本，同样需要 Resize，再提取 HOG 特征。

正样本给定的标签为 $1$，负样本给定的标签为 $0$。

训练时，基本根据论文中的参数设置，论文中 Resize 的尺寸为 $(32, 100)$，因此最后的特征维数为 $7200$。

$RandomForestClassifier$ 的 n_estimators（即决策树的个数）设置为 $10$，max_depth 最大深度为 $64$。

训练完成后，采用十折交叉验证。

**第二个阶段：过滤上一阶段的 Candidate Bounding Boxes**

对于每一张图像，其对应的有一些 Candidate Bounding Boxes，将每一个 Bounding Box 对应的图像区域 Crop 出来。Resize 到 $(32, 100)$，进行 HOG 特征提取，将提取的特征用上一个阶段训练好的 Random Forest Classifier 进行 Predict。

如果预测为正样本的概率大于 $0.5$，则保留此 Bounding Box；否则，剔除掉。

经过这个阶段的处理，可以过滤掉一些错检 Boxes。

#### **程序实现**

这部分的实现用到了 Python 包： skimage 模块中的 $hog$ 函数：http://scikit-image.org/docs/dev/auto_examples/plot_hog.html

Random Forest Classifier 用到了 Python 中的机器学习库： [sickit-learn](https://github.com/scikit-learn/scikit-learn)

这部分的代码在：`randomForest.py` 中。

这部分也将最后数据做了保存，保存为 `boxes_edgeBoxes_RFC.pkl`，存储经过 Random Forest Classifier 后的 Boxes。

最后，Python 文件 `test_RFC.py`，将其在图像上画出。图像最后存储在 `./edge_boxes_with_python/Output/edge_bboxes_with_RFC` 文件夹中。



但是！不明白为什么，经过随机森林后的 Box，定位的效果不如上一步（Edge Boxes）中，仅仅做了 Structured Edges Detect，再过滤掉 $IoU \lt 0.5$ 的结果。 

我在这里做了很多次的实验，调整参数。但均是如此，我猜想是数据集的原因。



## Bounding Box Regression

最后一部分，进行 Bounding Box 的回归。

因为经过上面的步骤，最后剩下的 Bounding Boxes 都是与 Ground Truth 的 $IoU$ 大于 $0.5$ 的。将每一张图像中的 Bounding Boxes Crop 出来。以这个 Bounding Box 的中心为坐标轴，进行 **Inflated**。文中的 “膨胀” factor 为 2，如果原先的 Bounding Box 的 Top-Left 、Bottom-Right 坐标为 $(x_1,y_1,x_2,y_2)$，那么，inflated 后的坐标为：

$$\left( x_1- \frac{(x_2-x_1)}{4},y_1- \frac{(y_2-y_1)}{4},x_2+ \frac{(x_2-x_1)}{4},y2+ \frac{(y_2-y_1)}{4} \right)$$

当然，还有几种特殊情况，程序里有体现。

在将 inflated 后的区域图像 Crop 出来，Resize 到 $ 32\times100$ 作为训练样本。此时，这部分对应的 Ground Truth 的坐标即作为训练的 Label。损失函数为 $L_2 ＝\sum \|g(I_b,\Phi)-q(b_{gt})\|_2^2$ 。



网络结构比较简单，程序实现中，我调用 caffe 的 python 接口，见 python 文件：`create_net.py`：

```python
import os, sys
CAFFE_PATH = '/home/chenxp/caffe/python'
sys.path.append(CAFFE_PATH)
import caffe
from caffe import layers as L
from caffe import params as P

def Bounding_Box_Reg(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=64, pad=2, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=128, pad=2, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv3 = L.Convolution(n.pool2, kernel_size=3, num_output=256, pad=1, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv4 = L.Convolution(n.pool3, kernel_size=3, num_output=512, pad=1, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.conv4, in_place=True)
    n.pool4 = L.Pooling(n.conv4, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1   = L.InnerProduct(n.pool4, num_output=4000, weight_filler=dict(type='xavier'))
    n.dp1   = L.Dropout(n.ip1, dropout_ratio=0.5)
    n.ip2   = L.InnerProduct(n.ip1, num_output=4, weight_filler=dict(type='xavier'))
    n.loss  = L.EuclideanLoss(n.ip2, n.label)

    return n.to_proto()

with open('BBR_train.prototxt', 'w') as f:
    f.write(str(Bounding_Box_Reg('train.h5', 16)))

with open('BBR_test.prototxt', 'w') as f:
    f.write(str(Bounding_Box_Reg('test.h5', 16)))
```

生成了两个网络结构文件：`BBR_train.prototxt`、`BBR_test.prototxt`，再写一个 `BBR_solver.prototxt`，网络的准备工作完成了。

接下来是这部分最关键的，生成训练数据。这部分程序文件为：`BBR_CreateDataset_HDF5.py` 。

因为最后的 Ground Truth 的 label 实际上是一个向量，一般使用 Caffe 的 LMDB 数据库组织数据只能是纯量 label。这里，我使用了 HDF5 数据格式文件来组织存放数据级 label。

最后生成了 HDF5 格式的数据，`train.h5`、`test.h5`，以及这个数据库的 TXT 路径文件（在 caffe 中直接读取 HDF5 会产生错误）：`train.txt`、`test.txt`。

最后这个文件夹`Bounding_Box_Reg`中，还有两个文件：`test_hdf5.py`，这是我测试 HDF5 数据的文件，`train.py`，这是训练网络文件。里面有设置 GPU，Solver 等。

看一轮训练结果：`solver.step(1)`

![](https://github.com/chenxinpeng/Reading-Text-in-the-Wild-with-Convolutional-Neural-Networks/blob/master/Bounding_Box_Reg/train_1.png)

还在训练中：

![](https://github.com/chenxinpeng/Reading-Text-in-the-Wild-with-Convolutional-Neural-Networks/blob/master/Bounding_Box_Reg/train_2.png)

因为是实验，数据集也小，训练参数也有待调试，就没全跑完，就到这吧。
