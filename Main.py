# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/27

from FCS import Fcs
from tkinter import filedialog
import pandas as pd
import numpy as np
import os, re, sys, warnings
from marker_ratio_calculation import markerRatioCalculation
from immuneAgeSubsets import subsetsRatioCalculation
from confidenceCalculation import confidence_calculation
from immuneAgeCalculation import predict_age
from immuneImpairment import immuneImpairmentMatrix


warnings.filterwarnings('ignore')



if __name__ == '__main__':
    ###################################################
    ####    1.读取singlets导出的FCS文件并转换成CSV    ####
    ###################################################
    # 选择路径
    Fpath = filedialog.askdirectory()
    os.makedirs(Fpath+"/WriteFcs/")
    os.makedirs(Fpath+"/Output/")
    csv_path = Fpath + '/WriteFcs/'
    output_path = Fpath + '/Output/'

    # 读取panel表信息
    panel_file = Fpath + "/panel.xlsx"
    panel_tuple = Fcs.export_panel_tuple(panel_file)
    print(panel_tuple)

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        # 重写marker-name
        pars = fcs.marker_rename(fcs.pars, *panel_tuple)
        new_filename = re.sub('-', '', filename)
        new_filename = re.sub('^.+?_', 'PLT_', new_filename)
        new_filename = re.sub(r'fcs$', 'csv', new_filename)
        new_file = Fpath + "/WriteFcs/" + new_filename
        fcs.write_to(new_file, pars, to="csv")
    print('所有样本FCS文件均已转换为CSV文件', '\n','-'*100,'\n')


    ###################################################
    ####           2.计算        ####
    ###################################################
    file_list = os.listdir(csv_path)
    ratio_all = pd.DataFrame()
    confidence_all = pd.DataFrame()
    immune_age_all = pd.DataFrame()
    ratio34_all = pd.DataFrame()
    impair_all = pd.DataFrame()

    for info in file_list:
        sample_id = info[:4]
        os.makedirs(output_path + '/%s/'%info[:4])
        sample_path = output_path + '/%s/'%info[:4]
        sample_df = pd.read_csv(csv_path+info).iloc[:, :-1]

        # 1. 计算各个marker的标签矩阵
        label_df = markerRatioCalculation(sample_df)
        print('Marker ratio calculation has finished!', '\n')

        # 2. 计算特定亚群的比率
        ratio_merge_df = subsetsRatioCalculation(label_df)
        ratio_df = ratio_merge_df.T
        ratio_df['subset'] = list(ratio_df.index)
        ratio_df.columns = ['frequency', 'subset']
        ratio_df.index = [i for i in range(ratio_df.shape[0])]
        ratio_df = ratio_df[['subset', 'frequency']]
        ratio_df['frequency'] = ratio_df['frequency'].apply(lambda x: x*100)
        print('Subset ratio calculation has finished!', '\n')
        ratio_df.to_excel(sample_path+'subset_ratio.xlsx', index=False)
        ratio_merge_df.index = [sample_id]
        ratio_all = ratio_all.append(ratio_merge_df)

        # 3. 计算置信区间相对值
        confidence_df = confidence_calculation(ratio_df)
        confidence_df.to_excel(sample_path+'confidence.xlsx', index=False)
        print('Confidence calculation has finished!', '\n')
        confidence_merge_df = confidence_df.T
        confidence_merge_df.columns = list(confidence_df['subset'].values)
        confidence_merge_df = confidence_merge_df.iloc[1:, :]
        confidence_merge_df.index = [sample_id]
        confidence_all = confidence_all.append(confidence_merge_df)

        # 4. 计算免疫年龄
        age_df, ratio34_df = predict_age(ratio_df)
        frequency_list = [info, 'x']
        frequency_list.extend(list(ratio34_df['frequency'].values))
        col_names = ['Patient.ID', 'age', 'B cells', 'CD161+CD4+ T cells', 'CD161+CD8+ T cells', 'CD161-CD45RA+ Tregs', 'CD161-CD45RA- Tregs', 'CD20- CD3- lymphocytes', 'CD28+CD8+ T cells', 'CD4+ T cells', 'CD4+CD28+ T cells', 'CD8+ T cells', 'CD85j+CD8+ T cells', 'IgD+CD27+ B cells', 'IgD+CD27- B cells', 'IgD-CD27+ B cells', 'IgD-CD27- B cells', 'NK cells', 'NKT cells', 'T cells', 'Tregs', 'central memory CD4+ T cells', 'central memory CD8+ T cells', 'effector CD4+ T cells', 'effector CD8+ T cells', 'effector memory CD4+ T cells', 'effector memory CD8+ T cells', 'gamma-delta T cells', 'lymphocytes', 'memory B cells', 'monocytes', 'naive B cells', 'naive CD4+ T cells', 'naive CD8+ T cells', 'transitional B cells', 'viable/singlets']
        ratio34_df = pd.DataFrame(frequency_list).T
        ratio34_df.columns = col_names
        age_df.index = [sample_id]
        age_df.columns = ['immune age']
        age_df.to_excel(sample_path+'immune_age.xlsx', index=False)
        ratio34_df.to_excel(sample_path+'subset34_ratio.xlsx', index=False)
        print('Immune age calculation has finished!', '\n')
        immune_age_all = immune_age_all.append(age_df)
        ratio34_all = ratio34_all.append(ratio34_df)

        # 5. 提取免疫损伤矩阵
        impair_df = immuneImpairmentMatrix(ratio_df, sample_id)
        print('Immune impairment matrix has finished!', '\n')
        impair_all = impair_all.append(impair_df)


    ratio_all.to_excel(output_path+'ratio_all.xlsx')
    confidence_all.to_excel(output_path+'confidence_all.xlsx')
    immune_age_all.to_excel(output_path+'immune_age_all.xlsx')
    ratio34_all.to_excel(output_path+'ratio34_all.xlsx', index=False)
    impair_all.to_excel(output_path+'impairment_all.xlsx', index=False)











