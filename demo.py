# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/5


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,fcluster,fclusterdata,dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing

cell = pd.read_csv('C:\\Users\\pc\\Desktop\\CellFrequency.csv')
print([column for column in cell])   # 列名 cluster名
print(cell.values)
# 预处理
min_max_scaler = preprocessing.MinMaxScaler()
data_M = min_max_scaler.fit_transform(cell.values.T)
print(data_M)

plt.figure(figsize=(20,12))
Z = linkage(data_M,method='ward',metric='euclidean')
p = dendrogram(Z,0)
plt.show()