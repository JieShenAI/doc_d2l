# softmax

在这一章的代码中

* 有计算分类精确度的函数

* 有 Accumulator 类

  

只关心，正确分类类别

## 损失函数

接下来，我们需要一个损失函数来度量预测的效果。
我们将使用最大似然估计，这与在线性回归
（ :numref:`subsec_normal_distribution_and_squared_loss`）
中的方法相同



# softmax回归
## 分类问题
:label:`subsec_classification-problem`

我们从一个图像分类问题开始。
假设每次输入是一个$2\times2$的灰度图像。
我们可以用一个标量表示每个像素值，每个图像对应四个特征$x_1, x_2, x_3, x_4$。
此外，假设每个图像属于类别“猫”，“鸡”和“狗”中的一个。

统计学家很早以前就发明了一种表示分类数据的简单方法：*独热编码*（one-hot encoding）。
独热编码是一个向量，它的分量和类别一样多。
类别对应的分量设置为1，其他所有分量设置为0。
在我们的例子中，标签$y$将是一个三维向量，
其中$(1, 0, 0)$对应于“猫”、$(0, 1, 0)$对应于“鸡”、$(0, 0, 1)$对应于“狗”：

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$



## 损失



`loss = nn.CrossEntropyLoss(reduction='none')`



在`train_epoch_ch3()`中:



* `l = loss(y_hat, y)`

  `y_hat = net(X)`

  

  y 不是 ont-hot编码的, 而是如下格式

  ```python
  tensor([8, 3, 7, 2, 7, 8, 3, 4, 3, 0, 8, 6, 3, 7, 8, 2, 7, 3, 7, 8, 8, 9, 8, 4,...])
  ```

  

* `l.mean().backward()`

  softmax的损失需要求均值之后，再反向传播



