import numpy as np
import pandas as pd

# 构造数据
def dataset():
    col1 = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    col2 = [3, 1, 7, np.nan, 4, 0, 5, 7, 12, np.nan]
    col3 = [3, np.nan, np.nan, np.nan, 9, np.nan, 10, np.nan, 4, np.nan]
    y = [10, 15, 8, 12, 17, 9, 7, 14, 16, 20]
    data = {'feature1':col1, 'feature2':col2, 'feature3':col3, 'label':y}
    df = pd.DataFrame(data)
    return df

# data = dataset()
# print(data)