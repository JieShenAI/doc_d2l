# 层和块

为了实现复杂的模型引入神经网络块的概念。

块由类表示。

## 自定义块

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```



* 继承 `nn.Module`
* 定义 `forward()` 前向传播函数、



## 顺序块



```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

`__init__`函数将每个模块逐个添加到有序字典`_modules`中。

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```



当然，除了在 `forward()`写网络layer，在其中也是可以执行操作的；



## 混搭



```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```


## 嵌套块

多个块嵌套，参数命名约定



```
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```



> Sequential(  (0): 
>
> Sequential(    
>
> (block 0): Sequential(      (0): Linear(in_features=4, out_features=8, bias=True)      (1): ReLU()      (2): Linear(in_features=8, out_features=4, bias=True)      (3): ReLU()    )    (block 1): Sequential(      (0): Linear(in_features=4, out_features=8, bias=True)      (1): ReLU()      (2): Linear(in_features=8, out_features=4, bias=True)      (3): ReLU()    )    
>
> (block 2): Sequential(      (0): Linear(in_features=4, out_features=8, bias=True)      (1): ReLU()      (2): Linear(in_features=8, out_features=4, bias=True)      (3): ReLU()    )    
>
> (block 3): Sequential(      (0): Linear(in_features=4, out_features=8, bias=True)      (1): ReLU()      (2): Linear(in_features=8, out_features=4, bias=True)      (3): ReLU()
>
> ...
>
> ​    )  )  
>
> (1): Linear(in_features=4, out_features=1, bias=True) )



层是分层嵌套，可通过嵌套列表索引访问；比如：

```
rgnet[0][1][0].bias.data
```



## 自定义层



### 不带参数的层

构建一个没有任何参数的层，



```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```



### 带参数的层



```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```



* `linear = MyLinear(5, 3)`
* `linear(torch.rand(2, 5))`
