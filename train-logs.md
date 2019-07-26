# 训练情况记录

## thchs30 batch_size=32

```
data_length = None:
训练20epochs，loss从220多降到70多，但是val_loss从230多，一直升到350多

data_length = 10000 // batch_size
训练10个epoch，loss从340降到202，val_loss开始269，到第5个epoch时达到最低的226，后面基本没怎么变化

data_length = None:
训练10个epoch，两个dropout比例从0.2改为0.5
Epoch 1/10: loss: 218.4021 - val_loss: 234.1096
Epoch 10/10: loss: 171.9361 - val_loss: 247.0067
loss下降很慢，val_loss缓慢上升
```



## 环境

2080 TI * 2
