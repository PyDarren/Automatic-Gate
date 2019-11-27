# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/27



import pandas as pd
import numpy as np


def func(subset, names, raw_confidence):
    vals = list(subset.values)
    names = list(names.values)
    arrows_95 = list()
    for i in range(len(vals)):
        name = names[i]
        reference = raw_confidence[raw_confidence['subset']==name]
        if vals[i] > reference['upper_95'].values[0]:
            arrows_95.append('↑')
        elif vals[i] < reference['low_95'].values[0]:
            arrows_95.append('↓')
        else:
            arrows_95.append(' ')
    sample_df = pd.DataFrame([names, arrows_95]).T
    sample_df.columns = ['subset', 'confidence_95']
    return sample_df



def confidence_calculation(df):
    path = 'C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/'
    raw_confidence = pd.read_excel(path+'Rawdata/confidence_output.xlsx').iloc[:, :4]
    select_subsets_df = pd.read_excel(path+'Rawdata/置信区间选择.xlsx')
    select_subsets = list(select_subsets_df['subset'].values)

    raw_confidence = raw_confidence[raw_confidence['subset'].isin(select_subsets)]
    df = df[df['subset'].isin(select_subsets)]
    subset = df['frequency']
    names = df['subset']
    sample_df = func(subset, names, raw_confidence)
    return sample_df





    # names = list(df.iloc[:, 0].values)
    #
    # for i in range(1, df.shape[1]):
    #     subset = df.iloc[:, i]
    #     sample_name = df.columns[i]
    #     sample_df = func(subset, names, raw_confidence, sample_name)
    #     new_confidence_df = pd.merge(new_confidence_df, sample_df, on='subset')
    #
    # columns_id = [0, 1]
    # columns_id.extend([i for i in range(3, new_confidence_df.shape[1]) if i % 2 != 0])
    # final_df = new_confidence_df.iloc[:, columns_id]
    # final_df = final_df[final_df['subset'].isin(select_subsets)]
    # final_df = pd.merge(select_subsets_df, final_df, on='subset')




