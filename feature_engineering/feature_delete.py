import data_construction


# 删除属性
def delete_feature(df):
    N = df.shape[0]  # 样本数
    no_nan_count = df.count().to_frame().T  # 每一维特征非缺失值的数量
    del_feature, save_feature = [], []
    for col in no_nan_count.columns.tolist():
        loss_rate = (N - no_nan_count[col].values[0]) / N  # 缺失率
        # print(loss_rate)
        if loss_rate > 0.5:  # 缺失率大于 50% 时，将这一维特征删除
            del_feature.append(col)
        else:
            save_feature.append(col)
    return del_feature, df[save_feature]


# data = data_construction.dataset()
# del_feature, df11 = delete_feature(data)
# print(data)
# print(del_feature)
# print(df11)
