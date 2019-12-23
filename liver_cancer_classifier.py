# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/20

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def liver_cancer_ratio(df):
    new_df = pd.DataFrame(df['frequency'].values).T
    new_df.columns = list(df['subset'].values)
    selected_subsets = ['Lymphocytes/CD3+', 'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+', 'Lymphocytes/CD3+/CD8+/HLA-DR+', 'Lymphocytes/CD3+/CD8+/PD1+',
                        'Lymphocytes/CD3-', 'Myeloid cells/CD56-CD14-/DC cells/mDC', 'Myeloid cells/HLA-DR-/MDSC',
                        'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                        'Lymphocytes/CD3-/B cells/Q2: 145Nd_IgD+ , 153Eu_CD27+',
                        'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-', 'Lymphocytes/NKT']
    coefs = [0.3395892606722482, -1.3187903342176324, -0.0020243658869819586, 0.2433488814614238, 4.982475377097407,
             -0.6511358674239148, 2.0574280637690534, 2.0180734385478027, -0.5127602757483701, -0.9520728164099216,
             -3.6069461748531797, 0.13596659507497838, -0.25279200499189763]
    intercept = 0.0
    new_frequency = new_df.loc[:, selected_subsets].values[0]
    file_name = 'subsets_model'
    data_path = 'C:/Users/pc/OneDrive/PLTTECH/Project/20191218_liver_cancer/rawdata/%s' % file_name + '.xlsx'
    raw_df = pd.read_excel(data_path)
    raw_df.iloc[:, 1:-1] = raw_df.iloc[:, 1:-1].multiply(100)
    raw_df = raw_df.loc[:, selected_subsets]
    stdsc = StandardScaler()
    raw_df = pd.DataFrame(stdsc.fit_transform(raw_df.values), columns=raw_df.columns)
    new_frequency_std = stdsc.transform(new_frequency.reshape(1, -1))
    z = intercept + np.dot(coefs, new_frequency_std[0])
    probability = 1 / (1 + np.exp(-z))
    return probability
