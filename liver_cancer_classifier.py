# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/20


import pandas as pd
import numpy as np


def liver_cancer_ratio(df):
    new_df = pd.DataFrame(df['frequency'].values).T
    new_df.columns = list(df['subset'].values)
    selected_subsets = ['Lymphocytes/CD3+/CD4+/PD1+', 'Lymphocytes/CD3+/CD8+',
                        'Lymphocytes/CD3+/CD8+/PD1+', 'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                        'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-', 'Lymphocytes/CD3-',
                        'Lymphocytes/CD3-/B cells/CD24+CD38+', 'Lymphocytes/CD3-/NK cells', 'Monocytes/CD14+CD16-',
                        'Myeloid cells/CD56-CD14-/DC cells/mDC']
    coefs = [0.09310552121571801, -0.077000242273637, 0.07005583854851068, -0.06406542969575807, -0.0217563497333608,
             -0.02670823997421349, 0.08813100832661719, -0.0628721466515184, 0.060633311956386014, 0.010976109262905225]
    intercept = 0.0
    new_frequency = new_df.loc[:, selected_subsets].values[0]
    z = intercept + np.dot(coefs, new_frequency)
    probability = 1 / (1 + np.exp(-z))
    return probability
