做法是先将数据集分成train_data和all test_data

然后将train_data划分成train_data_tr_attack，train_data_te_attack

将all_test_data划分成ref_data，test_data

假设攻击者已知train_data_tr_attack和ref_data的成员标签，并利用这部分数据的输出训练攻击模型

之后在train_data_te_attack和test_data上进行MIA攻击的评估



使用方法为

```
python train.py
python eval.py
```

train.py进行数据划分和模型训练，eval.py进行训练后模型的测试与MIA攻击

代码主要参照https://github.com/inspire-group/MIAdefenseSELENA