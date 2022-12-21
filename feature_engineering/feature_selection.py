# 加载 IRIS 数据集做演示
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = sns.load_dataset('iris')
print(df.shape)
df['species'] = LabelEncoder().fit_transform(df.iloc[:, 4])
df.head()


# 方差选择法
# 自己手写理论公式来实现功能
def VarianceThreshold(df, threshold=0.):
    dfc = df.iloc[:, :4].copy()
    print('>>>特征名：\n', dfc.columns.tolist())
    # 1 求方差
    var = np.sum(np.power(np.matrix(dfc.values) - np.matrix(dfc.mean()), 2), axis=0) / (dfc.shape[0] - 1)
    T = []
    # 2 筛选大于阈值的特征
    for index, v in enumerate(var.reshape(-1, 1)):
        if v > threshold:
            T.append(index)
    dfc = dfc.iloc[:, T]
    return var, dfc


# 阈值设置为 0.6
var, dfc = VarianceThreshold(df, 0.60)
print('\n>>>原始特征对应的方差值：\n', var)
print('\n>>>方差阈值选择后的特征名：\n', dfc.columns)
dfc.head()

# 调用 sklearn 模块 API 接口
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(0.6)
x_vt = vt.fit_transform(df.iloc[:, :4])
print(vt.variances_)
print(x_vt[:5])


# 相关系数--特征与特征
# 自己手写理论公式实现功能
def corr_selector(df):
    dfc = df.copy().iloc[:, :4]
    CORR = np.zeros((dfc.shape[1], dfc.shape[1]))
    delete, save = [], []
    for i in range(dfc.shape[1]):
        if dfc.columns.tolist()[i] not in delete:
            save.append(dfc.columns.tolist()[i])
        for j in range(i + 1, dfc.shape[1]):
            # 计算特征与特征之间的相关系数
            cov = np.sum((dfc.iloc[:, i] - dfc.iloc[:, i].mean()) * (df.iloc[:, j] - df.iloc[:, j].mean()))
            std = np.sqrt(np.sum((df.iloc[:, i] - df.iloc[:, i].mean()) ** 2)) * np.sqrt(
                np.sum((df.iloc[:, j] - df.iloc[:, j].mean()) ** 2))
            corr = cov / std
            CORR[i][j] = corr
            # 筛选掉高线性相关两两特征中的某一个特征
            if (np.abs(corr) > 0.89) and (dfc.columns.tolist()[j] not in delete):
                delete.append(dfc.columns.tolist()[j])
    dfc_ = dfc[save].copy()
    return CORR, dfc_


corr, dfc_ = corr_selector(df)
print(corr)  # 只算了一半
print(dfc_.head())


# 相关系数--特征与目标变量
# 自己手写理论公式实现功能
def corr_selector(df):
    X, y = df.iloc[:, :4], df.iloc[:, 4]
    cor_list = []
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    print(X.columns.tolist())
    print(cor_list)
    return cor_list


corr_selector(df)
# df.plot()
plt.savefig('corr.png')
# plt.show()

# 调用 sklearn 模块 API 接口
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from numpy import array

fun = lambda X, Y: tuple(array(list(map(lambda x: pearsonr(x.tolist(), Y.tolist()), X.T))).T)
sb = SelectKBest(fun, k=2)
x_sb = sb.fit_transform(df.iloc[:, :4], df.iloc[:, 4])
print('>>>检验统计值(相关系数)：\n', sb.scores_)
print('\n>>>P值：\n', sb.pvalues_)
x_sb[:5]

# 卡方检验
# 调用 sklearn 模块 API 接口
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

skb = SelectKBest(chi2, k=2)
x_skb = skb.fit_transform(df.iloc[:, :4], df.iloc[:, 4])
print('>>>检验统计值(卡方值)：\n', skb.scores_)
print('\n>>>P值：\n', skb.pvalues_)
x_skb[:5]


# 互信息法
# 自己手写理论公式实现功能
# np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y。
# numpy.intersect1d(ar1, ar2, assume_unique=False) [资源]
# 找到两个数组的交集。
# 返回两个输入数组中已排序的唯一值。
# math.log(x[, base])
# x -- 数值表达式。
# base -- 可选，底数，默认为 e
def MI_and_NMI():
    import math
    from sklearn import metrics
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    B = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    N, A_ids, B_ids = len(A), set(A), set(B)
    # print(N, A_ids, B_ids)

    # 互信息计算
    MI, eps = 0, 1.4e-45
    # 你说的这种应该是为了避免log0的结果
    # 所以添加一个非常小的数字,避免无穷的...log(0)的情况
    for i in A_ids:
        for j in B_ids:
            ida = np.where(A == i)  # 返回索引
            idb = np.where(B == j)
            idab = np.intersect1d(ida, idb)  # 返回相同的部分
            # print(ida,idb,idab)

            # 概率值
            px = 1.0 * len(ida[0]) / N  # 出现的次数/总样本数
            py = 1.0 * len(idb[0]) / N
            pxy = 1.0 * len(idab) / N
            # MI 值
            MI += pxy * math.log(pxy / (px * py) + eps, 2)

    # 标准互信息计算
    Hx = 0
    for i in A_ids:
        ida = np.where(A == i)
        px = 1.0 * len(ida[0]) / N
        Hx -= px * math.log(px + eps, 2)
    Hy = 0
    for j in B_ids:
        idb = np.where(B == j)
        py = 1.0 * len(idb[0]) / N
        Hy -= py * math.log(py + eps, 2)
    NMI = 2.0 * MI / (Hx + Hy)

    return MI, NMI, metrics.normalized_mutual_info_score(A, B)


MI, NMI, normalized_mutual_info_score = MI_and_NMI()
print('互信息,标准互信息及其得分', MI, NMI, normalized_mutual_info_score)

# 调用 sklearn 模块 API 接口
from sklearn.feature_selection import SelectKBest
from minepy import MINE


# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，
# 返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


# fun = lambda X, Y: list(map(lambda x:mic(x, Y), X.T))
fun = lambda X, Y: tuple(array(list(map(lambda x: mic(x.tolist(), Y.tolist()), X.T))).T)

# 选择K个最好的特征，返回特征选择后的数据
skb_ = SelectKBest(fun, k=2)
x_skb_ = skb_.fit_transform(df.iloc[:, :4], df.iloc[:, 4])
print('>>>检验统计值(互信息)：\n', skb_.scores_)
print('\n>>>P值：\n', skb_.pvalues_)
x_skb_[:5]

# # Wrapper 包装法
# 稳定性选择(Stability Selection)
# from sklearn.linear_model import RandomizedLasso, LinearRegression
# from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
# import warnings
#
# warnings.filterwarnings('ignore')
#
# X = df.iloc[:, :4]
# Y = df.iloc[:, 4]
# names = X.columns.tolist()
# print(names)
#
# # -------------------------------------
# # 回归
# rlasso = RandomizedLasso(alpha=0.025)
# rlasso.fit(X, Y)
#
# # -------------------------------------
# # 分类
# rlogistic = RandomizedLogisticRegression()
# rlogistic.fit(X, Y)
#
# print("\n>>>回归 Features sorted by their score:")
# print(rlasso.scores_)
# print(sorted(zip(map(lambda x: format(x, '.4f'), rlasso.scores_), names), reverse=True))
#
# print("\n>>>分类 Features sorted by their score:")
# print(rlogistic.scores_)
# print(sorted(zip(map(lambda x: format(x, '.4f'), rlogistic.scores_), names), reverse=True))
#
# lr = LinearRegression()
# lr.fit(X, Y)
# lr.coef_, lr.intercept_

# 递归特征消除
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

# use linear regression as the model
lr = LinearRegression()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=2)
rfe.fit(X, Y)

print("Features sorted by their rank:")
print(sorted(zip(rfe.ranking_, names)))

# Embedded 嵌入法
# 线性模型
from sklearn.linear_model import LinearRegression

X = df.iloc[:, :4]
Y = df.iloc[:, 4]
lr = LinearRegression()
lr.fit(X, Y)
print(X.columns.tolist())
print(lr.coef_)

# 正则化
# L1范数
from sklearn.linear_model import Lasso

X = df.iloc[:, :4]
Y = df.iloc[:, 4]
lasso = Lasso(alpha=0.3)
lasso.fit(X, Y)
print(X.columns.tolist())
print(lasso.coef_)

# L2范数
from sklearn.linear_model import Ridge

X = df.iloc[:, :4]
Y = df.iloc[:, 4]
ridge = Ridge(alpha=0.3)
ridge.fit(X, Y)
print(X.columns.tolist())
print(ridge.coef_)

# 树模型-待定

# 类别标签不平衡处理
# 欠采样：easyEnsemble
# 过采样：Synthetic Minority Over-Sampling Technique :SMOTE
# 加权处理

# 数据降维
def pca(df, k):
    X = np.mat(df.iloc[:, :-1])
    # 1 中心化
    X_mean = X - np.mean(X, axis=0)
    # 2 求协方差
    cov = np.cov(X_mean.T)
    # 3 求特征值和特征向量
    w, v = np.linalg.eig(cov)
    # 4 对特征值排序并提取前k个主成分所对应的特征向量
    w_ = np.argsort(w)[::-1]
    v_ = v[:, w_[:k]]
    # 5 将原数据映射相乘到新的特征向量中
    newF = X_mean * v_
    return newF, w, v


newF, w, v = pca(df, k=3)


def best_k(w):
    wSum = np.sum(w)
    comsum_rate, goal_rate, count = 0, 0.98, 0
    for k in range(len(w)):
        CR = w[k] / wSum  # 计算贡献率
        print(CR)
        comsum_rate += CR  # 计算累加贡献率
        count += 1
        if comsum_rate >= goal_rate:
            print('Best k .... 累加贡献率为：', comsum_rate, end='')
            return count


best_k(w)

# 贡献率累加曲线
def CRplot(w):
    wSum = np.sum(w)
    comsum_rate, L = 0, []
    for k in range(len(w)):
        CR = w[k]/wSum  # 计算贡献率
        comsum_rate += CR  # 计算累加贡献率
        L.append(comsum_rate)
    plt.plot(range(1,5,1), L)
    # plt.show()
CRplot(w)
