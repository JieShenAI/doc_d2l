# 线性代数



torch.Tensor的复制是 `xxx.clone()`



```python
x = torch.arange(12, dtype=torch.float32).reshape(3,4)
```

```
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
```



## 常见的方法

假设A是tensor

> A.mean()

| 方法     | 描述     |
| -------- | -------- |
| sum()    | 求和     |
| mean()   | 均值     |
| var()    | 方差     |
| numel()  | 元素个数 |
| cumsum() | 累计总和 |
| norm()   | 范数     |



* axis

  * A.mean(axis=0)

* sum / mean

  * 按指定维度求和/均值

    A.sum(axis=0)；A.mean(axis=0)

  * sum求和，保留维度

    A.sum(axis=1, keepdims=True)

* cumsum

  `x.cumsum(axis=0)`
  
  ```
  tensor([[ 0.,  1.,  2.,  3.],
          [ 4.,  6.,  8., 10.],
          [12., 15., 18., 21.]])
  ```
  
  当前行的值，会逐渐传递给下一行去累加。

## 降维

可以[**指定张量沿哪一个轴来通过求和降低维度**]。以矩阵为例，为了通过求和所有行的元素来降维（轴0），我们可以在调用函数时指定`axis=0`。

```
x.sum(axis=0)
```

```
tensor([12., 15., 18., 21.])
```



同样的`axis=1`将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。



## 积

### 点积

Dot Product

给定两个向量$\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，它们的**点积**（dot product）$\mathbf{x}^\top\mathbf{y}$

（或$\langle\mathbf{x},\mathbf{y}\rangle$）是相同位置的按元素乘积的**和**：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

```
a = torch.Tensor([1,2,3])
b = torch.Tensor([2,3,4])

torch.dot(a,b)
# or
a.dot(b)

# tensor(20.)
```



### 矩阵向量积

矩阵和向量的乘积

```
a = torch.arange(9,dtype=torch.float32).reshape(3,3)
b = torch.Tensor([0,1,0])
```

`a,b`

```
(tensor([[0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.]]),
 tensor([0., 1., 0.]))
```

>  mv: matrix vector

```
torch.mv(a,b)
```

```
tensor([1., 4., 7.])
```



### 矩阵乘法

`torch.mm()`



## 范数

`norm`

一个向量的范数，告诉我们一个向量有多大；不是维度，而是分量的大小；


第一个性质是：如果我们按常数因子$\alpha$缩放向量的所有元素，
其范数也会按相同常数因子的*绝对值*缩放：

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

第二个性质是我们熟悉的三角不等式:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

第三个性质简单地说范数必须是非负的:

$$f(\mathbf{x}) \geq 0.$$

这是有道理的。因为在大多数情况下，任何东西的最小的*大小*是0。
最后一个性质要求范数最小为0，当且仅当向量全由0组成。




假设$n$维向量$\mathbf{x}$中的元素是$x_1,\ldots,x_n$，其[**$L_2$*范数*是向量元素平方和的平方根：**]


(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)

其中，在$L_2$范数中常常省略下标$2$，也就是说$\|\mathbf{x}\|$等同于$\|\mathbf{x}\|_2$。

在代码中，我们可以按如下方式计算向量的$L_2$范数。



```
u = torch.tensor([3.0, -4.0])
torch.norm(u)

# 5
```



### L1范数

在深度学习中，我们更经常地使用$L_2$范数的平方。
你还会经常遇到[**$L_1$范数，它表示为向量元素的绝对值之和：**]

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

与$L_2$范数相比，$L_1$范数受异常值的影响较小。
为了计算$L_1$范数，我们将绝对值函数和按元素求和组合起来。

```
torch.abs(u).sum()
```



### Lp范数

$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

类似于向量的$L_2$范数，[**矩阵**]$\mathbf{X} \in \mathbb{R}^{m \times n}$(**的*Frobenius范数*（Frobenius norm）是矩阵元素平方和的平方根：**)

(**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**)

Frobenius范数满足向量范数的所有性质，它就像是矩阵形向量的$L_2$范数。
调用以下函数将计算矩阵的Frobenius范数。





### 范数和目标
:label:`subsec_norms_and_objectives`

在深度学习中，我们经常试图解决优化问题：
*最大化*分配给观测数据的概率;
*最小化*预测和真实观测之间的距离。
用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。
目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

## 关于线性代数的更多信息

线性代数还有很多，其中很多数学对于机器学习非常有用。
例如，矩阵可以分解为因子，这些分解可以显示真实世界数据集中的低维结构。
机器学习的整个子领域都侧重于使用矩阵分解及其向高阶张量的泛化，来发现数据集中的结构并解决预测问题。
我们相信，一旦你开始动手尝试并在真实数据集上应用了有效的机器学习模型，你会更倾向于学习更多数学。

如果你渴望了解有关线性代数的更多信息，你可以参考[线性代数运算的在线附录](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)或其他优秀资源 :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`。

## 小结

* 标量、向量、矩阵和张量是线性代数中的基本数学对象。
* 向量泛化自标量，矩阵泛化自向量。
* 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。
* 一个张量可以通过`sum`和`mean`沿指定的轴降低维度。
* 两个矩阵的按元素乘法被称为他们的Hadamard积。它与矩阵乘法不同。
* 在深度学习中，我们经常使用范数，如$L_1$范数、$L_2$范数和Frobenius范数。
* 我们可以对标量、向量、矩阵和张量执行各种操作。

## 练习

1. 证明一个矩阵$\mathbf{A}$的转置的转置是$\mathbf{A}$，即$(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 给出两个矩阵$\mathbf{A}$和$\mathbf{B}$，证明“它们转置的和”等于“它们和的转置”，即$\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 给定任意方阵$\mathbf{A}$，$\mathbf{A} + \mathbf{A}^\top$总是对称的吗?为什么?
1. 我们在本节中定义了形状$(2,3,4)$的张量`X`。`len(X)`的输出结果是什么？
1. 对于任意形状的张量`X`,`len(X)`是否总是对应于`X`特定轴的长度?这个轴是什么?
1. 运行`A/A.sum(axis=1)`，看看会发生什么。你能分析原因吗？
1. 考虑一个具有形状$(2,3,4)$的张量，在轴0、1、2上的求和输出是什么形状?
1. 为`linalg.norm`函数提供3个或更多轴的张量，并观察其输出。对于任意形状的张量这个函数计算得到什么?