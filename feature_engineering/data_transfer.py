import seaborn as sns
import numpy as np

df = sns.load_dataset('iris')
print(df.shape)
print(df.head())


# 1.Min-max 区间缩放法-极差标准化
# 自己手写理论公式来实现功能
# 缩放到区间 [0 1]
def Min_max(df):
    x_minmax = []
    for item in df.columns.tolist()[:4]:
        MM = (df[item] - np.min(df[item])) / (np.max(df[item]) - np.min(df[item]))
        x_minmax.append(MM.values)
    return np.array(np.matrix(x_minmax).T)


df_minmax_scaler = Min_max(df)

# 直接调用 sklearn 模块的 API 接口
# 极差标准化(最大最小值标准化)
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
x_minmax_scaler = mms.fit_transform(df.iloc[:, :4])


# 2.MaxAbs 极大值标准化
# 自己手写理论公式来实现功能

def MaxAbs(df):
    x_maxabs = []
    for item in df.columns.tolist()[:4]:
        Max = np.max(np.abs(df[item]))
        MA = np.abs(df[item]) / Max
        x_maxabs.append(MA)
    return np.array(np.matrix(x_maxabs).T)


df_maxabs_scaler = MaxAbs(df)

# 直接调用 sklearn 模块的 API 接口
# 极大值标准化
from sklearn.preprocessing import MaxAbsScaler

mas = MaxAbsScaler()
x_maxabs_scaler = mas.fit_transform(df.iloc[:, :4])


# 3.z-score 标准差标准化
# 自己手写理论公式来实现功能
# 标准化之后均值为 0，标准差为 1
def z_score(df):
    N, x_z = df.shape[0], []
    for item in df.columns.tolist()[:4]:
        mean = np.sum(df[item]) / N
        std = np.sqrt(np.sum((df[item] - mean) ** 2) / N)
        Z = (df[item] - mean) / std
        x_z.append(Z)
    return np.array(np.matrix(x_z).T)


df_std_scaler = z_score(df)
# 直接调用 sklearn 模块的 API 接口
# 标准差标准化
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_std_scaler = ss.fit_transform(df.iloc[:, :4])


# 4.归一化---总和归一化
# 自己手写理论公式来实现功能
# 处理成所有数据和为1，权重处理
def feature_importance(df):
    x_sum_scaler = []
    for item in df.columns.tolist()[:4]:
        S = np.sum(df[item])
        FI = df[item] / S
        x_sum_scaler.append(FI)
    return np.array(np.matrix(x_sum_scaler).T)


df_sum_scaler = feature_importance(df)


# 非线性归一化
# 自己手写理论公式来实现功能
# sigmoid 函数归一化
def sigmoid(df):
    x_sigmoid = []
    for item in df.columns.tolist()[:4]:
        S = 1 / (1 + np.exp(-df[item]))
        x_sigmoid.append(S)
    return np.array(np.matrix(x_sigmoid).T)


df_sigmoid = sigmoid(df)


# 特征二值化
# 自己手写理论公式来实现功能
def Binarizer(ages):
    ages_binarizer = []
    print('>>>原始的定量数据\n', ages)
    for age in ages:
        if (age > 0) and (age <= 18):
            ages_binarizer.append(0)
        elif (age >= 19) and (age <= 40):
            ages_binarizer.append(1)
        elif (age >= 41) and (age <= 60):
            ages_binarizer.append(2)
    print('\n>>>特征二值化之后的定性数据\n', ages_binarizer)
    return ages_binarizer


ages = [4, 6, 56, 48, 10, 12, 15, 26, 20, 30, 34, 23, 38, 45, 41, 18]
ages_to_binarizer = Binarizer(ages)

# 直接调用 sklearn 模块的 API 接口
# binary 二值化
# 使用上面的 IRIS 数据集
from sklearn.preprocessing import Binarizer

# 阈值自定义为 3.0
# 大于阈值映射为 1，反之为 0
b = Binarizer(threshold=3.0)
x_binarizer = b.fit_transform(df.iloc[:, :4])


# 等宽分箱法
# 自己手写理论公式来实现功能
def equal_width_box(data):
    # 划分的等份数、储存等宽分箱离散后的数据
    k, data_width_box = 3, data
    # 分箱的宽度、区间起始值(最小值)、离散值
    width, start, value = (max(data) - min(data)) / k, min(data), list(range(1, k + 1))
    for i in range(1, k + 1):
        # 实现公式 [a+(k−1)∗width, a+k∗width]
        left = start + (i - 1) * width  # 左区间
        right = start + (i * width)  # 右区间
        print('第 %d 个区间：[%.2f, %.2f]' % (i, left, right))

        for j in range(len(data)):
            if (data[j] >= left) and (data[j] <= right):  # 判断是否属于 value[i] 区间
                data_width_box[j] = value[i - 1]

    return data_width_box


data = [4, 6, 56, 48, 10, 12, 15, 26, 20, 30, 34, 23, 38, 45, 41, 18]
data_to_box = equal_width_box(data)

# 聚类划分
import seaborn as sns
from sklearn.cluster import KMeans

data = sns.load_dataset('iris')
X = data.iloc[:, 1]
kmeans = KMeans(n_clusters=4)  # 离散为 4 等份
kmeans.fit_transform(np.array(X).reshape(-1, 1))  # 只取一个特征进行聚类离散化
print('>>>原始数据：', X.tolist())
print('>>>聚类离散后：', kmeans.labels_)

# 序号编码
# 自己手写理论公式实现功能(可优化)
import seaborn as sns


def LabelEncoding(df):
    x, dfc = 'species', df
    key = dfc[x].unique()  # 将唯一值作为关键字
    value = [i for i in range(len(key))]  # 键值
    Dict = dict(zip(key, value))  # 字典，即键值对
    for i in range(len(key)):
        for j in range(dfc.shape[0]):
            if key[i] == dfc[x][j]:
                dfc[x][j] = Dict[key[i]]
    dfc[x] = dfc[x].astype(np.float32)
    return dfc[x]


data = sns.load_dataset('iris')
df_le = LabelEncoding(data)

# 调用 sklearn 模块的 API 接口
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x_le = le.fit_transform(data['species'])

# 独热编码
# 自己手写理论实现功能
import seaborn as sns
import pandas as pd


def OneHotEncoding(df):
    x, dfc = 'species', df.copy()
    key = dfc[x].unique()  # (1)
    value = np.ones(len(key))  # (2)
    Dict = dict(zip(key, value))  # (3)
    v = np.zeros((dfc.shape[0], len(key)))  # (4)
    for i in range(len(key)):
        for j in range(dfc.shape[0]):
            if key[i] == dfc[x][j]:
                v[j][i] = Dict[key[i]]  # (5)
    dfv = pd.DataFrame(data=v, columns=['species_'] + key)
    return pd.concat([dfc.drop(x, axis=1), dfv], axis=1)


data = sns.load_dataset('iris')
ohe = OneHotEncoding(data)

# 调用 sklearn 模块的 API 接口
# 注意要先序号编码再独热哑编码

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def string_strip(x):
    return x.strip()  # 去除字符串周围的特殊字符(如：逗号、空格等)


ohe_data = np.array(list(map(string_strip, data['species'].tolist())))

le = LabelEncoder()
le_ohe_data = le.fit_transform(ohe_data)

ohe_ = OneHotEncoder()
x_ohe = ohe_.fit_transform(le_ohe_data.reshape(-1, 1)).toarray()
x_ohe2 = pd.DataFrame(data=x_ohe, columns=['species_'] + np.array(['setosa', 'virginica', 'virginica']))

x_ohe3 = pd.concat([data.drop('species', axis=1), x_ohe2], axis=1)
print(1)

# 二进制编码
# 利用二进制对ID进行哈希映射，最终得到0/1特征向量，且维数少于独热编码，节省了存储空间
