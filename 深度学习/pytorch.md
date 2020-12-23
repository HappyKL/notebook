## 流程

1.构建数据集

2.设计模型

3.构建损失和优化器

4.训练周期（**前馈求损失，反馈求梯度，更新参数**）



以y=wx+b为例，loss= (y_pred - y)^2

```python
import torch

# 准备数据
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# 设计模型
class LinearModel(torch.nn.Module):
  def __init__(self):
    super(LinearModel, self).__init__()
    self.linear = torch.nn.Linear(1,1)
  def forward(self,x):
    y_pred = self.linear(x) # 会调用__call__函数
    return y_pred
model = LInearModel()

# 构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

# 训练
for epoch in range(100):
  # forward
  y_pred = model(x_data) 
  loss = criterion(y_pred, y_data) 
  print(epoch, loss)
  
  # backward
  optimizer.zero_grad()
  loss.backward()
  
  # update
	optimizer.step()

# 打印训练出的参数
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
  
# 测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

```

