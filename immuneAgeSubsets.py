# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/21


import pandas as pd
import numpy as np
import os


class SubsetsRatio():

    def __init__(self, raw_df):
        self.raw_df = raw_df

    def marker2_num(self, marker1, marker2, label1, label2):
        '''
        计算2个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :return: 该2个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        return len(select_df)


    def marker3_num(self, marker1, marker2, marker3, label1, label2, label3):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        return len(select_df)


    def marker4_num(self, marker1, marker2, marker3, marker4, label1, label2, label3, label4):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param marker4: 第四个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :param label4: 第四个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        select_df = select_df[select_df[marker4] == label4]
        return len(select_df)


    def marker5_num(self, marker1, marker2, marker3, marker4, marker5, label1, label2, label3, label4, label5):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param marker4: 第四个marker的名字
        :param marker5: 第五个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :param label4: 第四个marker的类别
        :param marker5: 第五个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        select_df = select_df[select_df[marker4] == label4]
        select_df = select_df[select_df[marker5] == label5]
        return len(select_df)





if __name__ == '__main__':
    path =  'C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/label_prediction/test/'
    file_list = os.listdir(path)

    for info in file_list:
        df_test = pd.read_excel(path+info)

        subset_ratio = SubsetsRatio(df_test)

        lymphocytes_num = subset_ratio.marker2_num('CD33', 'CD14', 1, 1)
        lymphocytes_ratio = lymphocytes_num / len(df_test)
        print('-'*100)
        print('亚群lymphocytes的比率为', lymphocytes_ratio)
        print('-'*100)

        lymphocytes_CD3Pos_num = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 0)
        lymphocytes_CD3Pos_ratio = lymphocytes_CD3Pos_num / lymphocytes_num
        print('亚群Lymphocytes/CD3+的比率为', lymphocytes_CD3Pos_ratio)
        print('-'*100)

        lymphocytes_CD3Pos_CD4Pos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)
        lymphocytes_CD3Pos_CD4Pos_ratio = lymphocytes_CD3Pos_CD4Pos_num / lymphocytes_CD3Pos_num
        print('亚群Lymphocytes/CD3+/CD4+的比率为', lymphocytes_CD3Pos_CD4Pos_ratio)
        print('-'*100)

        lymphocytes_CD3Pos_CD4Pos_CD27Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD27', 1, 1, 0, 0, 0)
        lymphocytes_CD3Pos_CD4Pos_CD27Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD27Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD27+的比率为', lymphocytes_CD3Pos_CD4Pos_CD27Pos_ratio)
        print('-'*100)

