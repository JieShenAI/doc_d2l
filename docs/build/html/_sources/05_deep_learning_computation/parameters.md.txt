# 参数管理



## 参数访问

可通过索引访问模型的任意层

* `net[2].state_dict()`

  查看所有参数

  ```python
  OrderedDict([('weight', tensor([[ 0.3231, -0.3373,  0.1639, -0.3125,  0.0527, -0.2957,  0.0192,  0.0039]])), ('bias', tensor([-0.2930]))])
  ```
  
  `type(net[2].bias) `： `torch.nn.parameter.Parameter`
  
  `net[2].bias` : `tensor([-0.2930], requires_grad=True)`
  
  `net[2].bias.data` : `tensor([-0.2930])`
  
  > data 属性获取参数的数值



* 获取所有参数

  ```python
  print(*[(name, param.shape) for name, param in net.named_parameters()])
  ```

  

* 另一种访问参数形式

  `net.state_dict()['2.bias'].data`



## 嵌套块收集参数



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
>  )  )  
>
> (1): Linear(in_features=4, out_features=1, bias=True) )



层是分层嵌套，可通过嵌套列表索引访问；比如：

```
rgnet[0][1][0].bias.data
```



## 参数初始化

### 内置初始化

调用内置的初始化器。下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，

且将偏置参数设置为0。



```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
```



#### 初始化为给定常数

`nn.init.constant_(m.weight, 1)`



## xavier

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
```



## 自定义初始化

### [**自定义初始化**]

有时，深度学习框架没有提供我们需要的初始化方法。
在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$



```
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```



## 参数绑定

<font color="red">暂未学习</font>

有时我们希望在多个层间共享参数：

我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。



```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```



> 这个例子表明第三个和第五个神经网络层的参数是绑定的。
>
> 它们不仅值相等，而且由相同的张量表示。
>
> 因此，如果我们改变其中一个参数，另一个参数也会改变。
>
> 你可能会思考：当参数绑定时，梯度会发生什么情况？
>
> 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层
>
> （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
