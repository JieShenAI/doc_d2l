# 初始化参数



## 直接赋值

直接对神经网络的某个层赋值参数

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```



## 某类型layer赋值



```python
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        
net.apply(init_weights);
```

