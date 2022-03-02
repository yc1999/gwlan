# 2022.02.28

- [x] 减少训练集进行训练
  > 原来的训练集有4.5M的source-target pair，降低训练集的为原来的0.1倍。
- [ ] 为什么训练集的loss没有下降？
    
    1.使用一个example不断地训练，如果没有过拟合，那么就代表模型写得有问题。
    
    2.关闭dropout,查看是否会过拟合。

- [x] 增加log，checkpoint的目录位置; 

- [ ] 将Positional Encoder修改成register buffer；

- [x] 加入test部分的代码
  > 没有加预测输出文件；
- [x] Count average sequence len of the train set.
- [x] Use validation set to be the train set to test whether the code is well written. code past test, it's okay.
- [ ] add soft constraint in the model.(GRU, Chracter CNN etc.)
    > 德文一共有30个字母，但是还有标点符号什么的。。。所以还是根据整个vocab来构建token embedding；
- [ ] Change the learning rate factor according to the Transformer paper. Maybe it is not that important because cosine scheduler maybe better.
https://github.com/yc1999/gwlan/blob/c57b891f11ec06b94cb98ca409c0b4c2e906fde8/models/wpm_model.py#L105-L106