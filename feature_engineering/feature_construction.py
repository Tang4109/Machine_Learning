import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

df = sns.load_dataset('iris')
# 序号编码将字符串类别转为ID类别
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df = df.rename(columns={'species': 'labels'})
df = df[['labels', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
print(df.shape)
print(df.head())

# 单变量
# 计数特征
# 简单示例:统计单个变量数值次数作为新的特征
newF1 = df.groupby(['petal_width'])['petal_width'].count().to_frame().rename(
    columns={'petal_width': 'petal_width_count'}).reset_index()
df_newF1 = pd.merge(df, newF1, on=['petal_width'], how='inner')
print('>>>新构建的计数特征的唯一值数据：\n', df_newF1['petal_width_count'].unique())
print(df_newF1.head())

# 多变量
name = {'count': 'petal_width_count', 'min': 'petal_width_min',
        'max': 'petal_width_max', 'mean': 'petal_width_mean',
        'std': 'petal_width_std'}
# The aggregation is for each column
# 根据Dataframe的列'sepal_length'进行划分，再选择'petal_width'列进行聚合操作,产生多列,并存为新列名
newF2 = df.groupby(by=['sepal_length'])['petal_width'].agg(
    ['count', 'min', 'max', 'mean', 'std']).rename(
    columns=name).reset_index()
df_newF2 = pd.merge(df, newF2, on='sepal_length', how='inner')
# 由于聚合分组之后有一些样本的 std 会存在缺失值(组内样本只有1个，算std时分母为0)，所以统一填充为 0
df_newF2['petal_width_std'] = df_newF2['petal_width_std'].fillna(0)
print(df_newF2.head())
# df_newF2.columns.tolist()

# 验证新特征的表征能力以及有效性
from sklearn.svm import SVC

# 原数据特征表征能力
original_feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X, Y = df_newF2[original_feature], df_newF2['labels']
svc = SVC()
svc.fit(X, Y)
print('>>>原数据特征的表征能力得分：%.4f' % (svc.score(X, Y)), '\n')

# 单个新特征对应的表征能力
# 新特征的表征能力大小与其对应目标之间高度相关系数成正比
# 比如 mean 对应 labels 相关系数最大，所训练得出的 score 也是最大的
new_feature_test = ['petal_width_count', 'petal_width_min', 'petal_width_max', 'petal_width_mean', 'petal_width_std']
for col in new_feature_test:
    X, Y = df_newF2[[col]], df_newF2['labels']
    svc = SVC()
    svc.fit(X, Y)
    print('>>>新特征 %s 的表征能力得分：%.4f' % (col, svc.score(X, Y)))

# 多个新特征组合对应的表征能力
print()
for col2 in new_feature_test:
    merge_feature = original_feature + [col2]
    X, Y = df_newF2[merge_feature], df_newF2['labels']
    svc = SVC()
    svc.fit(X, Y)
    print('>>>原始特征组合新特征 %s 的表征能力得分：%.4f' % (col2, svc.score(X, Y)))

# 函数变换
X = np.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]])
X1 = X ** 2
print('>>>平方\n', X1)
X2 = np.sqrt(X)
print('>>>开平方\n', X2)
X3 = np.exp(X)
print('>>>指数\n', X3)
X4 = np.log(X)
print('>>>对数\n', X4)

fig = plt.figure()

# 时间序列常用方法
# 非平稳序列转换成平稳序列
# 对数&差分
y = np.array([1.3, 5.5, 3.3, 5.3, 3.4, 8.0, 6.6, 8.7, 6.8, 7.9])
x = list(range(1, y.shape[0] + 1))
# 假设这是一个时间序列图

axes_1 = fig.add_subplot(1, 3, 1)
axes_1.plot(x, y)
axes_1.set_title('original plot')
axes_1.set_xlabel('time')
# 对数
y_log = np.log(y)
# 假设这是一个时间序列图
axes_2 = fig.add_subplot(1, 3, 2)
axes_2.plot(x, y_log)
axes_2.set_title('original plot')
axes_2.set_xlabel('time')


# 差分
def diff(dataset):
    DIFF = []
    # 由于差分之后的数据比原数据少一个
    DIFF.append(dataset[0])
    for i in range(1, dataset.shape[0]):  # 1 次差分
        value = dataset[i] - dataset[i - 1]
        DIFF.append(value)
    for i in range(1, dataset.shape[0]):  # 2 次差分
        value = DIFF[i] - DIFF[i - 1]
        DIFF.append(value)
    x = list(range(1, len(DIFF) + 1))
    axes_3 = fig.add_subplot(1, 3, 3)
    axes_3.plot(x, DIFF)
    axes_3.set_title('biff after')
    axes_3.set_xlabel('time')
    return DIFF


DIFF = diff(y)
# plt.show()

# 多项式:作用是生成新特征
from sklearn.preprocessing import PolynomialFeatures

print('>>>原始数据\n', X)
ploy1 = PolynomialFeatures(1)
print('>>>1 次项\n', ploy1.fit_transform(X))
ploy2 = PolynomialFeatures(2)
print('>>>2 次项\n', ploy2.fit_transform(X))
ploy3 = PolynomialFeatures(3)
print('>>>3 次项\n', ploy3.fit_transform(X))
# 1,x1,x2,x3
