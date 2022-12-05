# 标准差法
import seaborn as sns
import numpy as np

def std_(df):
    item, N = 'sepal_length', df.shape[0]
    M = np.sum(df[item])/N
    # print(M,np.mean(df[item]))
    assert (M - np.mean(df[item]) < 10e-7), 'mean is error'
    # assert (M == np.mean(df[item])), 'mean is error'
    S = np.sqrt(np.sum((df[item]-M)**2)/N)
    L, R = M-3*S, M+3*S
    print('正常区间值为 [%.4f, %.4f]' % (L, R))
    return (L, R)

df = sns.load_dataset('iris')
(L, R) = std_(df)

# MAD法
def MAD(df):
    item, N = 'sepal_length', df.shape[0]
    M = np.median(df[item])
    A = np.sqrt(np.sum((df[item]-M)**2)/N)
    L, R = M - 3 * A, M + 3 * A
    print('正常区间值为 [%.4f, %.4f]' % (L, R))
    return (L, R)

(L, R) = MAD(df)