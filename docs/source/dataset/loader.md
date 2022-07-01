# loader

## DataLoader



```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
```



```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = d2l.normal(0, 1, (num_examples, len(w))) # (1000,2)
    y = d2l.matmul(X, w) + b # 1000
    y += d2l.normal(0, 0.01, y.shape) # add noise
    return X, d2l.reshape(y, (-1, 1))
    
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
   
batch_size = 10
data_iter = load_array((features, labels), batch_size)

```



获取第一项

`next(iter(data_iter))`





## 坑



```python
def generate_data(w,b,nums):
    """x*w+b"""
    # x:(1000,2)
    x = torch.normal(20,5,(nums,len(w)))
    y = torch.mv(x,w) + b
    y += torch.normal(0,0.01,y.shape)
    # 错误
    # return x,y
    # 正确
    return x,y.reshape(-1,1)
```

需要 `y.reshape(-1,1)`

报错信息如下:

> c:\Users\username\anaconda3\lib\site-packages\torch\nn\modules\loss.py:529: UserWarning: Using a target size (torch.Size([200])) that is different to the input size (torch.Size([200, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.  return F.mse_loss(input, target, reduction=self.reduction)

在`generate_data`生成labels数据的shape是torch.Size(200)，要把它reshape成`torch.Size([200, 1])`;

我使用`torch.Size(200)`,发现模型收敛之后的效果很差。直到reshape成`torch.Size([200, 1])`才取得了不错的拟合效果；

