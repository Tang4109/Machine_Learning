import data_construction, feature_delete


def delete_sample(df):
    df_ = df.dropna()  # 自动丢弃NaN
    return df_


data = data_construction.dataset()
# print(data)
del_feature, df11 = feature_delete.delete_feature(data)
# print(df11)
df12 = delete_sample(df11)
print(df12)
