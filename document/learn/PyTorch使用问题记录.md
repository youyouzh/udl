# PyTorch使用问题记录

## 常规报错

### 损失函数定义错误

报错：`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`。

其中文意思大致是该张量含有多个（1个以上不含1个）boolean值，是不明确的，即无法比较。

报错的原因可能是loss函数申明没有加括号：

```python 
loss_function=nn.MSELoss   #错误
loss_function=nn.MSELoss() #正确
```

下面讲述一下比较两个张量是否相等的方法，如果是比较两个张量是否完全一致，直接用equal()即可：

```python 
features = torch.zeros(8, 32)
print(features[1].equal(torch.zeros(32))) # True

# 如果是以下的or，and类似的操作，依旧会报上面的错误，因为无法判断超过1个boolean的情况
a = torch.zeros(2)
b = torch.ones(2)
print((a == b) or (a != b)) # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

### 张量梯度问题

报错：`RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`。

待转换类型的PyTorch Tensor变量带有梯度，直接将其转换为numpy数据将破坏计算图，因此numpy拒绝进行数据转换，实际上这是对开发者的一种提醒。如果自己在转换数据时不需要保留梯度信息，可以在变量转换之前添加detach()调用。

解决方案： `y.numpy()` ---> `y.detach().numpy()`

若是数据部署在GPU上时，则修改为： `y.cpu().numpy()` ---> `y.cpu().detach().numpy()`

或者复制一个张量： `torch.tensor(loss, requires_grad=False)`

### 梯度清零错误

调用`x.grad.data.zero_()`时报错：`AttributeError: 'NoneType' object has no attribute 'data'`。

在使用pytorch实现多项线性回归中，在grad更新时，每一次运算后都需要将上一次的梯度记录清空，运行时报错，grad没有data这个属性，

原因是，在系统将w的grad值初始化为none，第一次求梯度计算是在none值上进行报错，自然会没有data属性

修改方法：添加一个判断语句，检查由grad再进行运算。
