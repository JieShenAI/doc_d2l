# 加载和保存



保存

```python
x = torch.arange(4)
torch.save(x, 'x-file')
```



读取

```python
x2 = torch.load('x-file')
```



存储一个张量列表

```
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```



读取或写入字典

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```



## 加载和保存模型

保存模型

```python
torch.save(net.state_dict(), 'mlp.params')
```



读出模型

```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

> eval() 是什么？



讨论：https://discuss.d2l.ai/t/topic/1839

