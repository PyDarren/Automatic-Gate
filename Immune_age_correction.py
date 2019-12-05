# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/5



import pandas as pd
import numpy as np
import os, sys
from sklearn import linear_model


def predict_age(df):
    path = 'C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/'
    formula_lvs = pd.read_csv(path+'Rawdata/formula_LVs.csv')
    subsets_34 = ['Lymphocytes/CD3-/B cells',
                  'Lymphocytes/CD3+/CD4+/CD161+',
                  'Lymphocytes/CD3+/CD8+/CD161+',
                  'Lymphocytes/CD3+/CD4+/Treg/Q1: 163Dy_CD161- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD4+/Treg/Q4: 163Dy_CD161- , 155Gd_CD45RA-',
                  'Lymphocytes/CD3-/CD3-CD20-',
                  'Lymphocytes/CD3+/CD8+/CD28+',
                  'Lymphocytes/CD3+/CD4+',
                  'Lymphocytes/CD3+/CD4+/CD28+',
                  'Lymphocytes/CD3+/CD8+',
                  'Lymphocytes/CD3+/CD8+/CD85j+',
                  'Lymphocytes/CD3-/B cells/Q2: 145Nd_IgD+ , 153Eu_CD27+',
                  'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-',
                  'Lymphocytes/CD3-/B cells/Q1: 145Nd_IgD- , 153Eu_CD27+',
                  'Lymphocytes/CD3-/B cells/Q4: 145Nd_IgD- , 153Eu_CD27-',
                  'Lymphocytes/CD3-/NK cells',
                  'Lymphocytes/NKT',
                  'Lymphocytes/CD3+',
                  'Lymphocytes/CD3+/CD4+/Treg',
                  'Lymphocytes/CD3+/CD4+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                  'Lymphocytes/gd T-cells',
                  'Lymphocytes',
                  'Lymphocytes/CD3-/B cells/CD24+CD38-',
                  'Monocytes',
                  'Lymphocytes/CD3-/B cells/CD24-CD38+',
                  'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                  'Lymphocytes/CD3-/B cells/CD24+CD38+',
                  'Singlets/Viable',
                  ]
    # df = df[df['subset'].isin(subsets_34)]
    # df.index = list(df['subset'].values)
    # df = df.loc[subsets_34]
    # df.index = [i for i in range(df.shape[0])]
    vals = df[subsets_34].values

    lv1 = np.sum(np.dot(vals, formula_lvs['LV1'].values))
    lv2 = np.sum(np.dot(vals, formula_lvs['LV2'].values))
    lv3 = np.sum(np.dot(vals, formula_lvs['LV3'].values))

    # immune_age = 40.1322 - 0.6259*lv1 + 0.2941*lv2 - 0.0356*lv3
    # immune_age = np.abs(immune_age)
    # print(immune_age)
    # age_df = pd.DataFrame([immune_age])
    # ratio34_df = df

    return lv1, lv2, lv3



if __name__ == '__main__':
    raw_df = pd.read_excel('E:/cd/Automatic_Gate_Data/zheyi_healthy/Result/免疫年龄修正.xlsx')

    lv1_list = list()
    lv2_list = list()
    lv3_list = list()

    for i in range(raw_df.shape[0]):
        df = raw_df.iloc[i, :]
        lv1, lv2, lv3 = predict_age(df)
        lv1_list.append(lv1)
        lv2_list.append(lv2)
        lv3_list.append(lv3)

    lv_df = pd.DataFrame([lv1_list, lv2_list, lv3_list]).T
    lv_df.columns = ['lv1', 'lv2', 'lv3']

    lv_vals = lv_df.values
    real_age = raw_df['age'].values

    reg = linear_model.LinearRegression()
    reg.fit(lv_vals, real_age)
    print(reg.coef_)