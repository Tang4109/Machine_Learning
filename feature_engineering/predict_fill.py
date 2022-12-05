import pandas as pd
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

dataset = sns.load_dataset('iris')

# (1)把需要填充缺失值的某一列特征(petal_width)作为新的标签(Label_petal_width)
# 将特征 petal_width 处理成含有 30 个缺失值的特征
dataset['Label_petal_length'] = dataset['petal_length']
# 每隔5个设为np.nan
true_test = []
for i in range(0, 150, 5):
    true_test.append(dataset.loc[i,'Label_petal_length'])
    dataset.loc[i, 'Label_petal_length'] = np.nan

# (2)然后找出与 Label_A 相关性较强的特征作为它的模型特征
# 可以发现特征 sepal_length、petal_width 与 Label_petal_width 有着强关联，
# 因此 sepal_length、petal_width 作为 Label_petal_length 的模型特征
# dataset.corr()#通过协方差矩阵来查看相关性

# (3)把 Label_petal_length 非缺失值部分作为训练集数据，而缺失值部分则作为测试集数据
data = dataset[['sepal_length', 'petal_width', 'Label_petal_length']].copy()  # 深拷贝与浅拷贝的问题
train = data[data['Label_petal_length'].notnull()]
test = data[data['Label_petal_length'].isnull()]
print(train.shape)
print(test.shape)

# (4)由于 Label_petal_length 的值属于连续型数值，则进行回归拟合
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 将训练集进行切分，方便验证模型训练的泛化能力
x_train, x_valid, y_train, y_valid = train_test_split(train.iloc[:, :2],
                                                      train.iloc[:, 2],
                                                      test_size=0.3
                                                      )
print(x_train.shape, x_valid.shape)
print(y_train.shape, y_valid.shape)

# 使用简单的线性回归进行训练
lr = LinearRegression()
lr.fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
print('>>>在训练集中的表现：', r2_score(y_train_pred, y_train))
y_valid_pred = lr.predict(x_valid)
print('>>>在验证集中的表现：', r2_score(y_valid_pred, y_valid))
print(np.array(y_valid))
print(y_valid_pred)

# (5)将训练学习到评分和泛化能力较好的模型去预测测试集，从而填充好缺失值
# 由上面来看，模型在训练集以及验证集上的表现相差不大并且效果挺不错的，
# 这说明模型的泛化能力不错，可以用于投放使用来预测测试集
y_test_pred = lr.predict(test.iloc[:, :2])
print('>>>在测试集中的表现：', r2_score(y_test_pred, np.array(true_test)))
test.loc[:, 'Label_petal_length'] = y_test_pred
df_no_nan = pd.concat([train, test], axis=0)  # axis=0表示纵向拼接
# print(df_no_nan.isnull().sum())
# print(df_no_nan.head())
print(np.array(true_test))
print(y_test_pred)