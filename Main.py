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

    for info in file_list:
        os.makedirs(output_path + '/%s/'%info[:4])
        sample_path = output_path + '/%s/'%info[:4]
        sample_df = pd.read_csv(csv_path+info).iloc[:, :-1]

        # 计算各个marker的标签矩阵
        label_df = markerRatioCalculation(sample_df)
        print('Marker ratio calculation has finished!', '\n')

        # 计算特定亚群的比率
        ratio_df = subsetsRatioCalculation(label_df).T
        ratio_df['subset'] = list(ratio_df.index)
        ratio_df.columns = ['frequency', 'subset']
        ratio_df.index = [i for i in range(ratio_df.shape[0])]
        ratio_df = ratio_df[['subset', 'frequency']]
        ratio_df['frequency'] = ratio_df['frequency'].apply(lambda x: x*100)
        print('Subset ratio calculation has finished!', '\n')
        ratio_df.to_excel(sample_path+'subset_ratio.xlsx', index=False)

        # 计算置信区间相对值
        confidence_df = confidence_calculation(ratio_df)
        confidence_df.to_excel(sample_path+'confidence.xlsx', index=False)












