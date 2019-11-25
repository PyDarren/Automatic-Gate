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


    def marker1_num(self, marker, label):
        select_df = self.raw_df[self.raw_df[marker] == label]
        return len(select_df), select_df


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
        return len(select_df), select_df


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
        return len(select_df), select_df


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
        return len(select_df), select_df


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
        return len(select_df), select_df


    def non_naive_cells(self, marker1='CD45', marker2='CD197', label1=0, label2=0):
        '''
        计算non naive cells个数
        :param marker1:
        :param marker2:
        :param label1:
        :param label2:
        :return:
        '''
        df = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[1]
        marker1_length = len(df[df[marker1] == label1])
        marker2_length = len(df[df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = df[df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = df[df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        non_naive_index = list(set(df.index).difference(select_df.index))
        non_naive = df.loc[non_naive_index, :]
        return len(non_naive), non_naive


    def cross_classification(self, marker1, marker2):
        '''
        计算十字分类后四个类别的个数
        :return:
        '''
        pos_and_pos = self.raw_df[np.logical_and(self.raw_df[marker1]==0, self.raw_df[marker2]==0)].shape[0]
        neg_and_neg = self.raw_df[np.logical_and(self.raw_df[marker1]==1, self.raw_df[marker2]==1)].shape[0]
        pos_and_neg = self.raw_df[np.logical_and(self.raw_df[marker1]==0, self.raw_df[marker2]==1)].shape[0]
        neg_and_pos = self.raw_df[np.logical_and(self.raw_df[marker1]==1, self.raw_df[marker2]==0)].shape[0]
        return pos_and_pos, neg_and_neg, pos_and_neg, neg_and_pos










if __name__ == '__main__':
    path =  'C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/label_prediction/test/'
    file_list = os.listdir(path)

    for info in file_list:
        df_test = pd.read_excel(path+info)

        subset_ratio = SubsetsRatio(df_test)

        # 3. lymphocytes
        lymphocytes_num = subset_ratio.marker2_num('CD33', 'CD14', 1, 1)[0]
        lymphocytes_ratio = lymphocytes_num / len(df_test)
        print('-'*100)
        print('亚群lymphocytes的比率为', lymphocytes_ratio)
        print('-'*100)

        # 4. Lymphocytes/CD3+
        lymphocytes_CD3Pos_num = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 0)[0]
        lymphocytes_CD3Pos_ratio = lymphocytes_CD3Pos_num / lymphocytes_num
        print('亚群Lymphocytes/CD3+的比率为', lymphocytes_CD3Pos_ratio)
        print('-'*100)

        # 5. Lymphocytes/CD3+/CD4+
        lymphocytes_CD3Pos_CD4Pos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_ratio = lymphocytes_CD3Pos_CD4Pos_num / lymphocytes_CD3Pos_num
        print('亚群Lymphocytes/CD3+/CD4+的比率为', lymphocytes_CD3Pos_CD4Pos_ratio)
        print('-'*100)

        # 6. Lymphocytes/CD3+/CD4+/CD27+
        lymphocytes_CD3Pos_CD4Pos_CD27Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD27', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD27Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD27Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD27+的比率为', lymphocytes_CD3Pos_CD4Pos_CD27Pos_ratio)
        print('-'*100)

        # 7. Lymphocytes/CD3+/CD4+/CD27-
        lymphocytes_CD3Pos_CD4Pos_CD27Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD27', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD4Pos_CD27Neg_ratio = lymphocytes_CD3Pos_CD4Pos_CD27Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD27-的比率为', lymphocytes_CD3Pos_CD4Pos_CD27Neg_ratio)
        print('-'*100)

        # 8. Lymphocytes/CD3+/CD4+/CD28+
        lymphocytes_CD3Pos_CD4Pos_CD28Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD28', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD28Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD28Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD28+的比率为', lymphocytes_CD3Pos_CD4Pos_CD28Pos_ratio)
        print('-'*100)

        # 9. Lymphocytes/CD3+/CD4+/CD28-
        lymphocytes_CD3Pos_CD4Pos_CD28Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD28', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD4Pos_CD28Neg_ratio = lymphocytes_CD3Pos_CD4Pos_CD28Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD28-的比率为', lymphocytes_CD3Pos_CD4Pos_CD28Neg_ratio)
        print('-'*100)

        # 10. Lymphocytes/CD3+/CD4+/CD57+
        lymphocytes_CD3Pos_CD4Pos_CD57Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD57', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD57Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD57Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD57+的比率为', lymphocytes_CD3Pos_CD4Pos_CD57Pos_ratio)
        print('-'*100)
        
        # 11. Lymphocytes/CD3+/CD4+/CD85j+
        lymphocytes_CD3Pos_CD4Pos_CD85jPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD85j', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD85jPos_ratio = lymphocytes_CD3Pos_CD4Pos_CD85jPos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD85j+的比率为', lymphocytes_CD3Pos_CD4Pos_CD85jPos_ratio)
        print('-'*100)
        
        # 12. Lymphocytes/CD3+/CD4+/CD85j-
        lymphocytes_CD3Pos_CD4Pos_CD85jNeg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD85j', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD4Pos_CD85jNeg_ratio = lymphocytes_CD3Pos_CD4Pos_CD85jNeg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD85j-的比率为', lymphocytes_CD3Pos_CD4Pos_CD85jNeg_ratio)
        print('-'*100)
        
        # 13. Lymphocytes/CD3+/CD4+/CD94+
        lymphocytes_CD3Pos_CD4Pos_CD94Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD94', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD94Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD94Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD94+的比率为', lymphocytes_CD3Pos_CD4Pos_CD94Pos_ratio)
        print('-'*100)

        # 14. Lymphocytes/CD3+/CD4+/CD94-
        lymphocytes_CD3Pos_CD4Pos_CD94Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD94', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD4Pos_CD94Neg_ratio = lymphocytes_CD3Pos_CD4Pos_CD94Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD94-的比率为', lymphocytes_CD3Pos_CD4Pos_CD94Neg_ratio)
        print('-'*100)

        # 15. Lymphocytes/CD3+/CD4+/CD161+
        lymphocytes_CD3Pos_CD4Pos_CD161Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD161', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD161Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD161Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD161+的比率为', lymphocytes_CD3Pos_CD4Pos_CD161Pos_ratio)
        print('-'*100)

        # 16. Lymphocytes/CD3+/CD4+/CD161-
        lymphocytes_CD3Pos_CD4Pos_CD161Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD161', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD4Pos_CD161Neg_ratio = lymphocytes_CD3Pos_CD4Pos_CD161Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD161-的比率为', lymphocytes_CD3Pos_CD4Pos_CD161Neg_ratio)
        print('-'*100)

        # 17. Lymphocytes/CD3+/CD4+/CD152+
        lymphocytes_CD3Pos_CD4Pos_CD152Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD152', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD152Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD152Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD152+的比率为', lymphocytes_CD3Pos_CD4Pos_CD152Pos_ratio)
        print('-'*100)
        
        # 18. Lymphocytes/CD3+/CD4+/HLA_DR+
        lymphocytes_CD3Pos_CD4Pos_HLA_DRPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'HLA_DR', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_HLA_DRPos_ratio = lymphocytes_CD3Pos_CD4Pos_HLA_DRPos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/HLA_DR+的比率为', lymphocytes_CD3Pos_CD4Pos_HLA_DRPos_ratio)
        print('-'*100)

        # 19. Lymphocytes/CD3+/CD4+/CD278+
        lymphocytes_CD3Pos_CD4Pos_CD278Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD278', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_CD278Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD278Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/CD278+的比率为', lymphocytes_CD3Pos_CD4Pos_CD278Pos_ratio)
        print('-'*100)

        # 20. Lymphocytes/CD3+/CD4+/non-naive cells
        lymphocytes_CD3Pos_CD4Pos_non_naive_num = subset_ratio.non_naive_cells()[0]
        lymphocytes_CD3Pos_CD4Pos_non_naive_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive cells的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_ratio)
        print('-'*100)

        # 21. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5+
        non_naive_df = subset_ratio.non_naive_cells()[1]
        nonNaive = SubsetsRatio(non_naive_df)
        lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_num = nonNaive.marker1_num('CXCR5', 0)[0]
        lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive/CXCR5+的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_ratio)
        print('-'*100)

        # 22. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-
        lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num = nonNaive.marker1_num('CXCR5', 1)[0]
        lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive/CXCR5-的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_ratio)
        print('-'*100)

        # 23. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6+
        CXCR5Neg_df = nonNaive.marker1_num('CXCR5', 1)[1]
        CXCR5Neg = SubsetsRatio(CXCR5Neg_df)
        CXCR3_CCR6 = CXCR5Neg.cross_classification('CD183', 'CCR6')
        CXCR3Pos_CCR6Pos_num = CXCR3_CCR6[0]
        CXCR3Pos_CCR6Pos_ratio = CXCR3Pos_CCR6Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6+的比率为', CXCR3Pos_CCR6Pos_ratio)
        print('-'*100)

        # 24. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6-
        CXCR3Pos_CCR6Neg_num = CXCR3_CCR6[2]
        CXCR3Pos_CCR6Neg_ratio = CXCR3Pos_CCR6Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6-的比率为', CXCR3Pos_CCR6Neg_ratio)
        print('-'*100)

        # 25. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6-
        CXCR3Neg_CCR6Neg_num = CXCR3_CCR6[1]
        CXCR3Neg_CCR6Neg_ratio = CXCR3Neg_CCR6Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6-的比率为', CXCR3Neg_CCR6Neg_ratio)
        print('-'*100)

        # 26. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6+
        CXCR3Neg_CCR6Pos_num = CXCR3_CCR6[3]
        CXCR3Neg_CCR6Pos_ratio = CXCR3Neg_CCR6Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
        print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6+的比率为', CXCR3Neg_CCR6Pos_ratio)
        print('-'*100)

        # 27. Lymphocytes/CD3+/CD4+/PD1+
        lymphocytes_CD3Pos_CD4Pos_PD1Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'PD1', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_PD1Pos_ratio = lymphocytes_CD3Pos_CD4Pos_PD1Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/PD1+的比率为', lymphocytes_CD3Pos_CD4Pos_PD1Pos_ratio)
        print('-'*100)

        # 28. Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+
        CD4Pos_df = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[1]
        CD4Pos = SubsetsRatio(CD4Pos_df)
        CD197_CD45 = CD4Pos.cross_classification('CD197', 'CD45')
        CD197Neg_CD45Pos_num = CD197_CD45[3]
        CD197Neg_CD45Pos_ratio = CD197Neg_CD45Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+的比率为', CD197Neg_CD45Pos_ratio)
        print('-'*100)

        # 29. Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+
        CD197Pos_CD45Pos_num = CD197_CD45[0]
        CD197Pos_CD45Pos_ratio = CD197Pos_CD45Pos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+的比率为', CD197Pos_CD45Pos_ratio)
        print('-'*100)

        # 30. Lymphocytes/CD3+/CD4+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-
        CD197Pos_CD45Neg_num = CD197_CD45[2]
        CD197Pos_CD45Neg_ratio = CD197Pos_CD45Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-的比率为', CD197Pos_CD45Neg_ratio)
        print('-'*100)

        # 31. Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-
        CD197Neg_CD45Neg_num = CD197_CD45[1]
        CD197Neg_CD45Neg_ratio = CD197Neg_CD45Neg_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA-的比率为', CD197Neg_CD45Neg_ratio)
        print('-'*100)

        # 32. Lymphocytes/CD3+/CD4+/Q5: 176Yb_HLA_DR- , 172Yb_CD38+
        # 33. Lymphocytes/CD3+/CD4+/Q6: 176Yb_HLA_DR+ , 172Yb_CD38+
        # 34. Lymphocytes/CD3+/CD4+/Q7: 176Yb_HLA_DR+ , 172Yb_CD38-
        # 35. Lymphocytes/CD3+/CD4+/Q8: 176Yb_HLA_DR- , 172Yb_CD38-

        # 36. Lymphocytes/CD3+/CD4+/T-bet+
        lymphocytes_CD3Pos_CD4Pos_tbetPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'tbet', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD4Pos_tbetPos_ratio = lymphocytes_CD3Pos_CD4Pos_tbetPos_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/tbet+的比率为', lymphocytes_CD3Pos_CD4Pos_tbetPos_ratio)
        print('-'*100)

        # 37. Lymphocytes/CD3+/CD4+/Tfh
        tfh_num = CD4Pos.marker2_num('CXCR5', 'PD1', 0, 0)[0]
        tfh_ratio = tfh_num / lymphocytes_CD3Pos_CD4Pos_num
        print('亚群Lymphocytes/CD3+/CD4+/Tfh的比率为', tfh_ratio)
        print('-'*100)

        # 38. Lymphocytes/CD3+/CD4+/Treg
        treg_num = CD4Pos.marker2_num()

        # 39. Lymphocytes/CD3+/CD4+/Treg/Q1: 163Dy_CD161- , 155Gd_CD45RA+
        # 40. Lymphocytes/CD3+/CD4+/Treg/Q2: 163Dy_CD161+ , 155Gd_CD45RA+
        # 41. Lymphocytes/CD3+/CD4+/Treg/Q3: 163Dy_CD161+ , 155Gd_CD45RA-
        # 42. Lymphocytes/CD3+/CD4+/Treg/Q4: 163Dy_CD161- , 155Gd_CD45RA-
        
        # 43. Lymphocytes/CD3+/CD8+
        lymphocytes_CD3Pos_CD8Pos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD8', 1, 1, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_ratio = lymphocytes_CD3Pos_CD8Pos_num / lymphocytes_CD3Pos_num
        print('亚群Lymphocytes/CD3+/CD8+的比率为', lymphocytes_CD3Pos_CD8Pos_ratio)
        print('-'*100)

        # 44. Lymphocytes/CD3+/CD8+/CD27+
        lymphocytes_CD3Pos_CD8Pos_CD27Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD27', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD27Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD27Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD27+的比率为', lymphocytes_CD3Pos_CD8Pos_CD27Pos_ratio)
        print('-'*100)

        # 45. Lymphocytes/CD3+/CD8+/CD27-
        lymphocytes_CD3Pos_CD8Pos_CD27Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD27', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD8Pos_CD27Neg_ratio = lymphocytes_CD3Pos_CD8Pos_CD27Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD27-的比率为', lymphocytes_CD3Pos_CD8Pos_CD27Neg_ratio)
        print('-'*100)

        # 46. Lymphocytes/CD3+/CD8+/CD28+
        lymphocytes_CD3Pos_CD8Pos_CD28Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD28', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD28Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD28Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD28+的比率为', lymphocytes_CD3Pos_CD8Pos_CD28Pos_ratio)
        print('-'*100)

        # 47. Lymphocytes/CD3+/CD8+/CD28-
        lymphocytes_CD3Pos_CD8Pos_CD28Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD28', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD8Pos_CD28Neg_ratio = lymphocytes_CD3Pos_CD8Pos_CD28Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD28-的比率为', lymphocytes_CD3Pos_CD8Pos_CD28Neg_ratio)
        print('-'*100)

        # 48. Lymphocytes/CD3+/CD8+/CD57+
        lymphocytes_CD3Pos_CD8Pos_CD57Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD57', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD57Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD57Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD57+的比率为', lymphocytes_CD3Pos_CD8Pos_CD57Pos_ratio)
        print('-'*100)
        
        # 49. Lymphocytes/CD3+/CD8+/CD85j+
        lymphocytes_CD3Pos_CD8Pos_CD85jPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD85j', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD85jPos_ratio = lymphocytes_CD3Pos_CD8Pos_CD85jPos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD85j+的比率为', lymphocytes_CD3Pos_CD8Pos_CD85jPos_ratio)
        print('-'*100)
        
        # 50. Lymphocytes/CD3+/CD8+/CD85j-
        lymphocytes_CD3Pos_CD8Pos_CD85jNeg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD85j', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD8Pos_CD85jNeg_ratio = lymphocytes_CD3Pos_CD8Pos_CD85jNeg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD85j-的比率为', lymphocytes_CD3Pos_CD8Pos_CD85jNeg_ratio)
        print('-'*100)
        
        # 51. Lymphocytes/CD3+/CD8+/CD94+
        lymphocytes_CD3Pos_CD8Pos_CD94Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD94', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD94Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD94Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD94+的比率为', lymphocytes_CD3Pos_CD8Pos_CD94Pos_ratio)
        print('-'*100)

        # 52. Lymphocytes/CD3+/CD8+/CD94-
        lymphocytes_CD3Pos_CD8Pos_CD94Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD94', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD8Pos_CD94Neg_ratio = lymphocytes_CD3Pos_CD8Pos_CD94Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD94-的比率为', lymphocytes_CD3Pos_CD8Pos_CD94Neg_ratio)
        print('-'*100)

        # 53. Lymphocytes/CD3+/CD8+/CD161+
        lymphocytes_CD3Pos_CD8Pos_CD161Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD161', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD161Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD161Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD161+的比率为', lymphocytes_CD3Pos_CD8Pos_CD161Pos_ratio)
        print('-'*100)

        # 54. Lymphocytes/CD3+/CD8+/CD161-
        lymphocytes_CD3Pos_CD8Pos_CD161Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD161', 1, 1, 0, 0, 1)[0]
        lymphocytes_CD3Pos_CD8Pos_CD161Neg_ratio = lymphocytes_CD3Pos_CD8Pos_CD161Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD161-的比率为', lymphocytes_CD3Pos_CD8Pos_CD161Neg_ratio)
        print('-'*100)
        
        # 55. Lymphocytes/CD3+/CD8+/CD152+
        lymphocytes_CD3Pos_CD8Pos_CD152Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD152', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD152Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD152Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD152+的比率为', lymphocytes_CD3Pos_CD8Pos_CD152Pos_ratio)
        print('-'*100)
        
        # 56. Lymphocytes/CD3+/CD8+/CXCR5+
        lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CXCR5', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CXCR5+的比率为', lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_ratio)
        print('-'*100)
        
        # 57. Lymphocytes/CD3+/CD8+/granzyme_B+
        lymphocytes_CD3Pos_CD8Pos_granzyme_BPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'granzyme_B', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_granzyme_BPos_ratio = lymphocytes_CD3Pos_CD8Pos_granzyme_BPos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/granzyme_B+的比率为', lymphocytes_CD3Pos_CD8Pos_granzyme_BPos_ratio)
        print('-'*100)

        # 58. Lymphocytes/CD3+/CD8+/HLA_DR+
        lymphocytes_CD3Pos_CD8Pos_HLA_DRPos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'HLA_DR', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_HLA_DRPos_ratio = lymphocytes_CD3Pos_CD8Pos_HLA_DRPos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/HLA_DR+的比率为', lymphocytes_CD3Pos_CD8Pos_HLA_DRPos_ratio)
        print('-'*100)

        # 59. Lymphocytes/CD3+/CD8+/CD278+
        lymphocytes_CD3Pos_CD8Pos_CD278Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD278', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_CD278Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD278Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/CD278+的比率为', lymphocytes_CD3Pos_CD8Pos_CD278Pos_ratio)
        print('-'*100)

        # 60. Lymphocytes/CD3+/CD8+/PD1+
        lymphocytes_CD3Pos_CD8Pos_PD1Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'PD1', 1, 1, 0, 0, 0)[0]
        lymphocytes_CD3Pos_CD8Pos_PD1Pos_ratio = lymphocytes_CD3Pos_CD8Pos_PD1Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/PD1+的比率为', lymphocytes_CD3Pos_CD8Pos_PD1Pos_ratio)
        print('-'*100)

        # 61. Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+
        CD8Pos_df = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD8', 1, 1, 0, 0)[1]
        CD8Pos = SubsetsRatio(CD8Pos_df)
        CD197_CD45 = CD8Pos.cross_classification('CD197', 'CD45')
        CD197Neg_CD45Pos_num = CD197_CD45[3]
        CD197Neg_CD45Pos_ratio = CD197Neg_CD45Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+的比率为', CD197Neg_CD45Pos_ratio)
        print('-'*100)

        # 62. Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+
        CD197Pos_CD45Pos_num = CD197_CD45[0]
        CD197Pos_CD45Pos_ratio = CD197Pos_CD45Pos_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+的比率为', CD197Pos_CD45Pos_ratio)
        print('-'*100)

        # 63. Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-
        CD197Pos_CD45Neg_num = CD197_CD45[2]
        CD197Pos_CD45Neg_ratio = CD197Pos_CD45Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-的比率为', CD197Pos_CD45Neg_ratio)
        print('-'*100)

        # 64. Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-
        CD197Neg_CD45Neg_num = CD197_CD45[1]
        CD197Neg_CD45Neg_ratio = CD197Neg_CD45Neg_num / lymphocytes_CD3Pos_CD8Pos_num
        print('亚群Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA-的比率为', CD197Neg_CD45Neg_ratio)
        print('-'*100)

        # 65. Lymphocytes/CD3+/CD8+/Q5: 176Yb_HLA_DR- , 172Yb_CD38+
        # 66. Lymphocytes/CD3+/CD8+/Q6: 176Yb_HLA_DR+ , 172Yb_CD38+
        # 67. Lymphocytes/CD3+/CD8+/Q7: 176Yb_HLA_DR+ , 172Yb_CD38-
        # 68. Lymphocytes/CD3+/CD8+/Q8: 176Yb_HLA_DR- , 172Yb_CD38-

        # 69. Lymphocytes/CD3+/HLA-DR+
        lymphocytes_CD3Pos_HLA_DRPos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'HLA_DR', 1, 1, 0, 0)[0]
        lymphocytes_CD3Pos_HLA_DRPos_ratio = lymphocytes_CD3Pos_HLA_DRPos_num / lymphocytes_CD3Pos_num
        print('亚群Lymphocytes/CD3+/HLA_DR+的比率为', lymphocytes_CD3Pos_HLA_DRPos_ratio)
        print('-'*100)

        # 70. Lymphocytes/CD3-
        lymphocytes_CD3Neg_num = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 1)[0]
        lymphocytes_CD3Neg_ratio = lymphocytes_CD3Neg_num / lymphocytes_num
        print('亚群Lymphocytes/CD3-的比率为', lymphocytes_CD3Neg_ratio)
        print('-'*100)

        # 71. Lymphocytes/CD3-/B cells
        CD3Neg_df = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 1)[1]
        CD3Neg = SubsetsRatio(CD3Neg_df)
        CD3Neg_CD20Neg_num = CD3Neg.marker2_num('CD20', 'CD19', 0, 0)[0]
        CD3Neg_CD20Neg_ratio = CD3Neg_CD20Neg_num / lymphocytes_CD3Neg_num
        print('亚群CD3Neg_CD20Neg_ratio的比率为', CD3Neg_CD20Neg_ratio)
        print('-'*100)

        # 72. Lymphocytes/CD3-/B cells /CD24+CD38+
        # 73. Lymphocytes/CD3-/B cells /CD24+CD38-
        # 74. Lymphocytes/CD3-/B cells /CD24-CD38+

        # 75. Lymphocytes/CD3-/B cells /Q1: 145Nd_IgD- , 153Eu_CD27+
        CD3Neg_CD20Neg_df = CD3Neg.marker2_num('CD20', 'CD19', 0, 0)[1]
        CD3Neg_CD20Neg = SubsetsRatio(CD3Neg_CD20Neg_df)
        IGD_CD27 = CD3Neg_CD20Neg.cross_classification('IGD', 'CD27')
        IGDNeg_CD27Pos_num = IGD_CD27[3]
        IGDNeg_CD27Pos_ratio = IGDNeg_CD27Pos_num / CD3Neg_CD20Neg_num
        print('亚群Lymphocytes/CD3-/B cells /Q1: 145Nd_IgD- , 153Eu_CD27+的比率为', IGDNeg_CD27Pos_ratio)
        print('-'*100)

        # 76. Lymphocytes/CD3-/B cells /Q2: 145Nd_IgD+ , 153Eu_CD27+
        IGDPos_CD27Pos_num = IGD_CD27[0]
        IGDPos_CD27Pos_ratio = IGDPos_CD27Pos_num / CD3Neg_CD20Neg_num
        print('亚群Lymphocytes/CD3-/B cells /Q1: 145Nd_IgD+ , 153Eu_CD27+的比率为', IGDPos_CD27Pos_ratio)
        print('-'*100)

        # 77. Lymphocytes/CD3-/B cells /Q3: 145Nd_IgD+ , 153Eu_CD27-
        IGDPos_CD27Neg_num = IGD_CD27[2]
        IGDPos_CD27Neg_ratio = IGDPos_CD27Neg_num / CD3Neg_CD20Neg_num
        print('亚群Lymphocytes/CD3-/B cells /Q1: 145Nd_IgD+ , 153Eu_CD27-的比率为', IGDPos_CD27Neg_ratio)
        print('-'*100)

        # 78. Lymphocytes/CD3-/B cells /Q4: 145Nd_IgD- , 153Eu_CD27-
        IGDNeg_CD27Neg_num = IGD_CD27[1]
        IGDNeg_CD27Neg_ratio = IGDNeg_CD27Neg_num / CD3Neg_CD20Neg_num
        print('亚群Lymphocytes/CD3-/B cells /Q1: 145Nd_IgD- , 153Eu_CD27-的比率为', IGDNeg_CD27Neg_ratio)
        print('-'*100)

        # 79. Lymphocytes/CD3-/CD3-CD20-
        CD3Neg_df = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 1)[1]
        CD3Neg = SubsetsRatio(CD3Neg_df)
        CD3Neg_CD20Neg_num = CD3Neg.marker2_num('CD20', 'CD19', 1, 1)[0]
        CD3Neg_CD20Neg_ratio = CD3Neg_CD20Neg_num / lymphocytes_CD3Neg_num
        print('亚群CD3Neg_CD20Neg_ratio的比率为', CD3Neg_CD20Neg_ratio)
        print('-'*100)

        # 80. Lymphocytes/CD3-/CD3-CD20-/Plasmablasts



