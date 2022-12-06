# 标准差法
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def std_(df):
    item, N = 'sepal_length', df.shape[0]
    M = np.sum(df[item]) / N
    # print(M,np.mean(df[item]))
    assert (M - np.mean(df[item]) < 10e-7), 'mean is error'
    # assert (M == np.mean(df[item])), 'mean is error'
    S = np.sqrt(np.sum((df[item] - M) ** 2) / N)
    L, R = M - 3 * S, M + 3 * S
    print('正常区间值为 [%.4f, %.4f]' % (L, R))
    return (L, R)


df = sns.load_dataset('iris')
(L, R) = std_(df)


# MAD法
def MAD(df):
    item, N = 'sepal_length', df.shape[0]
    M = np.median(df[item])
    A = np.sqrt(np.sum((df[item] - M) ** 2) / N)
    L, R = M - 3 * A, M + 3 * A
    print('正常区间值为 [%.4f, %.4f]' % (L, R))
    return (L, R)


(L, R) = MAD(df)


# 箱形图法
def boxplot(data):
    # 下四分位数值、中位数，上四分位数值
    Q1, median, Q3 = np.percentile(data, (25, 50, 75), interpolation='midpoint')  # interpolation=插值
    # 四分位距
    IQR = Q3 - Q1

    # 内限
    inner = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    # 外限
    outer = [Q1 - 3.0 * IQR, Q3 + 3.0 * IQR]
    print('>>>内限：', inner)
    print('>>>外限：', outer)

    # 过滤掉极端异常值
    print(len(data))
    goodData = []
    for value in data:
        if (value < outer[1]) and (value > outer[0]):
            goodData.append(value)
    print(len(goodData))

    return goodData


data = [0.2, 0.3, 0.15, 0.32, 1.5, 0.17, 0.28, 4.3, 0.8, 0.43, 0.67]
boxplot(data)

# 图像对比法
# 功能实现
# 构造一个演示数据
D1 = {'feature1': [1, 4, 3, 4, 6.3, 6, 7, 8, 8.3, 10, 9.5, 12, 11.2, 14.5, 17.8, 15.3, 17.3, 17, 19, 18.8],
      'feature2': [11, 20, 38, 40, 59, 61, 77, 84, 99, 115, 123, 134, 130, 155, 138, 160, 152, 160, 189, 234],
      'label': [1, 5, 9, 4, 12, 6, 17, 25, 19, 10, 31, 11, 13, 21, 15, 28, 35, 24, 19, 20]}
D2 = {'feature1': [1, 3, 3, 6, 5, 6, 7, 10, 9, 10, 13, 12, 16, 14, 15, 16, 14, 21, 19, 20],
      'feature2': [13, 25, 33, 49, 45, 66, 74, 86, 92, 119, 127, 21, 13, 44, 34, 29, 168, 174, 178, 230]}
df_train = pd.DataFrame(data=D1)
df_test = pd.DataFrame(data=D2)
L = [df_train.iloc[:, 1], df_test.iloc[:, 1], 'train_feature2', 'test_feature2']

fig = plt.figure(figsize=(15, 5))
X = list(range(df_train.shape[0]))
for i in range(D2.__len__()):
    ax = fig.add_subplot(1, 2, i + 1)
    ax.plot(X, L[i], label=L[i + 2], color='red')
    ax.legend()
    ax.set_xlabel('Section')
plt.show()
