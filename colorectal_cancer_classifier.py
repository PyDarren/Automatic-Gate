# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/20

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def colorectal_cancer_ratio(df):
    new_df = pd.DataFrame(df['frequency'].values).T
    new_df.columns = list(df['subset'].values)
    selected_subsets = ['Lymphocytes/CD3+', 'Lymphocytes/CD3+/CD4+', 'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+/HLA-DR+', 'Lymphocytes/CD3+/CD8+/PD1+', 'Lymphocytes/CD3-',
                        'Myeloid cells/CD56-CD14-/DC cells/mDC', 'Myeloid cells/HLA-DR-/MDSC',
                        'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3-/B cells/Q2: 145Nd_IgD+ , 153Eu_CD27+',
                        'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-', 'Lymphocytes/NKT']
    coefs = [0.9074448021743782, 1.0390403595457338, -0.05740507577089001, 2.3332424335378428, 2.559187344993976,
             -0.40911363602478157, 0.8121301312191118, 2.620548182357151, 0.49525274872248637, -0.37292687375891687,
             -1.4375030761917278, -0.26236577541542033]
    intercept = 0.728162928081308
    new_frequency = new_df.loc[:, selected_subsets].values[0]
    file_name = 'rawdata'
    data_path = 'C:/Users/pc/OneDrive/PLTTECH/Project/20191217_colorectal_cancer/rawdata/%s' % file_name + '.xlsx'
    raw_df = pd.read_excel(data_path)
    raw_df.iloc[:, 1:-1] = raw_df.iloc[:, 1:-1].multiply(100)
    raw_df = raw_df.loc[:, selected_subsets]
    stdsc = StandardScaler()
    raw_df = pd.DataFrame(stdsc.fit_transform(raw_df.values), columns=raw_df.columns)
    new_frequency_std = stdsc.transform(new_frequency.reshape(1, -1))
    z = intercept + np.dot(coefs, new_frequency_std[0])
    probability = 1 / (1 + np.exp(-z))
    return probability
