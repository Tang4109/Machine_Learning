import numpy as np

import data_construction, feature_delete


def delete_sample(df):
    df_ = df.dropna()  # 自动丢弃NaN
    return df_


data = data_construction.dataset()
# print(data)
del_feature, df11 = feature_delete.delete_feature(data)
# print(df11)
# df12 = sample_delete.delete_sample(df11)

# 均值填充
print(df11.mean())
df13 = df11.fillna(df11.mean())
print('df13:\n', df13)

# 中位数填充
print(df11.median())
df14 = df11.fillna(df11.median())
print('df14:\n', df14)


# 众数填充
# print(df11.mode())
# 由于众数可能会存在多个，因此返回的是序列而不是一个值
# 所以在填充众数的时候，我们可以 df11['feature'].mode()[0]，可以取第一个众数作为填充值
def mode_fill(df):
    for col in df.columns.tolist():
        if df[col].isnull().sum() > 0:  # 有缺失值就进行众数填充
            df_ = df.fillna(df[col].mode()[0])

    return df_


df15 = mode_fill(df11)
print('df15:\n', df15)

# 最大值/最小值填充
df16 = df11.fillna(df11.max())
print('df16:\n', df16)
df17 = df11.fillna(df11.min())
print('df17:\n', df17)

# 统一值填充
# 自定义统一值常数为 10
df18 = df11.fillna(value=10)
print('df18:\n', df18)

# 前后向填充:反复进行前后向填充可以解决第一个和最后一个数据缺失值的问题
df19 = df11.fillna(method='ffill')  # 前向填充
print('df19:\n', df19)

df20 = df11.fillna(method='bfill')  # 后向填充
print('df20:\n', df20)


# 多项式插值填充:本质是一个多项式拟合
def Polynomial(x, y, test_x):
    '''
    test_x 的值一般是在缺失值的前几个或者后几个值当中，挑出一个作为参考值，
    将其值代入到插值模型之中，学习出一个值作为缺失值的填充值
    '''
    # 求待定系数
    array_x = np.array(x)  # 向量化
    array_y = np.array(y)
    n, X = len(x), []
    for i in range(n):  # 形成 X 矩阵
        l = array_x ** i
        X.append(l)
    X = np.array(X).T
    A = np.dot(np.linalg.inv(X), array_y.T)  # 根据公式求待定系数 A

    # 缺失值插值
    xx = []
    for j in range(n):
        k = test_x ** j
        xx.append(k)
    xx = np.array(xx)
    return np.dot(xx, A.T)


x, y, test_x = [1, 2, 3, 4], [1, 5, 2, 6], 3.5
Polynomial_value = Polynomial(x, y, test_x)
print('多项式拟合值:', Polynomial_value)


# lagrange插值
def Lagrange(x, y, test_x):
    '''
    所谓的插值法，就是在X范围区间中挑选一个或者自定义一个数值，
    然后代进去插值公式当中，求出数值作为缺失值的数据。
    '''
    n = len(x)
    L = 0
    for i in range(n):
        # 计算公式 1
        li = 1
        for j in range(n):
            if j != i:
                li *= (test_x - x[j]) / (x[i] - x[j])
        # 计算公式 2
        L += li * y[i]
    return L


Lagrange_value = Lagrange(x, y, test_x)
print('Lagrange拟合值:', Lagrange_value)
