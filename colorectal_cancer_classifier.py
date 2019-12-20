# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/20



import pandas as pd
import numpy as np


def colorectal_cancer_ratio(df):
    new_df = pd.DataFrame(df['frequency'].values).T
    new_df.columns = list(df['subset'].values)
    selected_subsets = ['Lymphocytes/CD3+/CD4+/PD1+', 'Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                    'Lymphocytes/CD3+/CD8+', 'Lymphocytes/CD3+/CD8+/CD28+', 'Lymphocytes/CD3+/CD8+/HLA-DR+',
                    'Lymphocytes/CD3+/CD8+/PD1+', 'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                    'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-', 'Lymphocytes/CD3-',
                    'Lymphocytes/CD3-/B cells', 'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-',
                    'Lymphocytes/CD3-/NK cells', 'Monocytes/CD14+CD16-', 'Myeloid cells/CD56-CD14-/DC cells/mDC',
                    'Myeloid cells/HLA-DR-/MDSC']
    coefs = [0.00927043629406046, -0.028946298340720118, -0.017782653637124574, 0.02585048206046803, 0.021593750484386377,
             0.0186779162603874, -0.01635284582746002, -0.0020207341380261445, -0.004452683853446408, -0.02075606413740891,
             -0.06474285237590295, -0.0023891468611911917, 0.057490858445839885, 0.015806718090745634, 0.04014936422387438]
    intercept = -0.0009541340919598513
    new_frequency = new_df.loc[:, selected_subsets].values[0]
    z = intercept + np.dot(coefs, new_frequency)
    probability = 1 / (1 + np.exp(-z))
    return probability
