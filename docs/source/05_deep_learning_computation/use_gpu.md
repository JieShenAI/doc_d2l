# GPU



存储在gpu上

`X = torch.ones(2, 3, device=try_gpu())`



查询可用gpu数量

```python
torch.cuda.device_count()
```



```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```



## 多GPU

在第２个gpu创建张量

```python
Y = torch.rand(2, 3, device=try_gpu(1))
```



X在第一个GPU 上。由于x和y不在同一个gpu上，故二者不能直接相加

将x移到gpu(1)上，才能和Y相加

```
Z = x.cuda(1)
Y + Z
```



## 模型放GPU

```
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```



## 扩展

python GPU运算的性能有一个绕不开的概念：`全局解释器锁`

