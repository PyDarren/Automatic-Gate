# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/20


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def lung_cancer_ratio(df):
    new_df = pd.DataFrame(df['frequency'].values).T
    new_df.columns = list(df['subset'].values)
    selected_subsets = ['Lymphocytes/CD3+', 'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+', 'Lymphocytes/CD3+/CD8+/HLA-DR+', 'Lymphocytes/CD3+/CD8+/PD1+',
                        'Lymphocytes/CD3-/B cells', 'Lymphocytes/CD3-/NK cells', 'Monocytes',
                        'Myeloid cells/CD56-CD14-/DC cells/mDC', 'Myeloid cells/CD56-CD14-/DC cells/pDC',
                        'Myeloid cells/HLA-DR-/MDSC', 'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                        'Lymphocytes/CD3-/B cells/Q2: 145Nd_IgD+ , 153Eu_CD27+',
                        'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-', 'Lymphocytes/gd T-cells',
                        'Lymphocytes/NKT']
    coefs = [-0.4726753039444568, -0.4910499419650651, -0.40788521500237934, 0.49591933930055904, 0.4463356060103869,
             0.24501605408765015, -1.0184137186386444, -0.02392357668960775, 1.315014370716627, -0.418727269716841,
             0.4269358605984087, -0.29338162466750434, 0.01828232826172561, -0.22676404157577304, 0.5503861965485389,
             -0.38621813894085905, 0.03380241831155741]
    intercept = -0.8695816855576141
    new_frequency = new_df.loc[:, selected_subsets].values[0]
    file_name = 'data_all'
    data_path = 'C:/Users/pc/OneDrive/PLTTECH/Project/20191205_lung_cancer/rawdata/%s' % file_name + '.xlsx'
    raw_df = pd.read_excel(data_path)
    raw_df.iloc[:, 1:-1] = raw_df.iloc[:, 1:-1].multiply(100)
    raw_df = raw_df.loc[:, selected_subsets]
    stdsc = StandardScaler()
    raw_df = pd.DataFrame(stdsc.fit_transform(raw_df.values), columns=raw_df.columns)
    new_frequency_std = stdsc.transform(new_frequency.reshape(1, -1))
    z = intercept + np.dot(coefs, new_frequency_std[0])
    probability = 1 / (1 + np.exp(-z))
    return probability