Relation-Extraction With Bert and graph

目前只做了 semeval 2018 task7

## 简介 
1. 实现了两种模型：Bert-sp w/ entity indicator （models/naive_bert.py）和一种用GNN增强的模型（models/graph_trans.py）
2. 目前做了semeval subtask 1.1 和 subtask 2

## 使用

### 训练 
见instructions.txt

## 复用

### 代码复用
1. 两种模型的定义在 models/naive_bert.py 和 models/graph_trans.py 中
2. 模型输入和输出：见 models/naive_bert.py 中的 forward 的注释

### 模型参数复用
我还没写保存参数的代码 \_(:з」∠)\_ ，需要训练之后的参数的话也可以直接在 train.py 的最后加一句 tc.save(trained_models)

## 注意事项
1. models/loss_func.py 中定义了几种 loss function ，调试之后发现 loss_3 是最好的（这也是默认的选择）。loss_3里面的 class_weight 是各个类别的权重，我是大致按训练集里面的比例的反比写死的（并且 no_rel 的权重是调参调出来的），如果要拿去做其他数据集的话需要改一下这些参数。

2. models/gene_func.py 是从模型预测结果生成答案文件的方法

3. dataloader.py 是读入 semeval 2018 task7 的方法，我是写了一个很蹩脚的直接基于规则解析字符串的方法... 这里面会丢掉一些标注得很不规范实在很难解析的训练数据（比如有嵌套的entity mention之类的）。不过测试数据保证不会扔，但是太长的测试数据会被截短（以满足 bert 的长度要求），超出的 entity 会被直接扔掉。事实上不会有太多数据和 entity 被扔掉。
（处理的时候会输出一些信息，只要没有 assertion error 就是没问题的）

4. 输入的时候如果加上指令--rel_only，表示没有 no_rel 这个type，也就是 subtask 1，否则是有 no_rel 这个 type 的，也就是 subtask 2。

5. 因为没有验证集，训练的时候输出的结果就是测试集的结果。也不根据这个结果来选择模型，就拿所有epoch跑完的模型参数作为最终的模型参数。

6. 训练的时候用了一些tricks，主要来自于《ETH-DS3Lab at SemEval-2018 Task 7: Effectively Combining Recurrent and Convolutional Neural Networks for Relation Classification and Extraction》，包括ensemble。训练时通过 --ensenble_size=x 来控制ensemble的模型个数，默认是5。


## 结果

（其中打星号的是使用了ensemble的方法。）

subtask 2:

Method                       | Macro-F1 | Micro-F1
-----------------------------|----------|-----------
* Rotsztejn et.al.           |  49.3    |  -
* naive_bert                 |  48.29   |  43.88
* graph_trans                |  50.44   |  45.02

subtasl 1.1:

Method                         | Macro-F1 | Micro-F1
-------------------------------|----------|-----------
* Rotsztejn et.al.             |  81.72   |  82.82
* naive_bert                   |  82.27   |  83.38
* graph_trans                  |  83.06   |  83.94
Wang et.al.(single-per-pass)   |  81.4    |  83.1
Wang et.al.(multiple-per-pass) |  80.5    |  83.9
naive_bert                     |  78.49   |  80.56
graph_trans                    |  80.57   |  81.13

其中 Rotsztejn et.al. 是在semeval 2018 task7的比赛上取得最高成绩的模型。

Wang et.al. 是《Extracting Multiple-Relations in One-Pass with Pre-Trained Transformers》这篇的模型。
 
