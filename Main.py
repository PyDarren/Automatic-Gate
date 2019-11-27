# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/27

from FCS import Fcs
from tkinter import filedialog
import pandas as pd
import numpy as np
import os, re, sys, warnings
from .marker_ratio_calculation import markerRatioCalculation


warnings.filterwarnings('ignore')



if __name__ == '__main__':
    ###################################################
    ####    1.读取singlets导出的FCS文件并转换成CSV    ####
    ###################################################
    # 选择路径
    Fpath = filedialog.askdirectory()
    os.makedirs(Fpath+"/WriteFcs/")
    csv_path = Fpath + '/WriteFcs/'

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
    print('所有样本FCS文件均已转换为CSV文件', '-'*100,'\n')


    ###################################################
    ####           2.计算所有marker的标签矩阵        ####
    ###################################################
    file_list = os.listdir(csv_path)

    for info in file_list:
        sample_df = pd.read_csv(csv_path+info)
        label_df = markerRatioCalculation(sample_df)













