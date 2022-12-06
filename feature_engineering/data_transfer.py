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
        MM = (df[item] - np.min(df[item]))/(np.max(df[item])-np.min(df[item]))
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
        MA = np.abs(df[item])/Max
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
        mean = np.sum(df[item])/N
        std = np.sqrt(np.sum((df[item]-mean)**2)/N)
        Z = (df[item] - mean)/std
        x_z.append(Z)
    return np.array(np.matrix(x_z).T)[:5]
z_score(df)
