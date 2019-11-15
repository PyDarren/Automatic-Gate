# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/14


import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys, warnings

warnings.filterwarnings(action='ignore')


def ratioCalculation2(df, model):
    '''
    计算二分类模型的亚群比率
    :param df:
    :return:
    '''
    test = df.values
    predictions = model.predict(test)
    pre_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
    pre_0_length = len([i for i in pre_labels if i == 0])
    pre_1_length = len([i for i in pre_labels if i == 1])
    length_df = df.shape[0]
    df['class'] = pre_labels
    sub_df = df[df['class']==0]
    ratio_0 = pre_0_length / length_df * 100
    ratio_1 = pre_1_length / length_df * 100
    ratio_list = [ratio_0, ratio_1]
    return ratio_list, sub_df



if __name__ == "__main__":

    #### Load Models
    input_shape = (None, 47)

    model_CD3 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD3_classfy.h5')
    model_CD3.build(input_shape)
    
    model_CD4 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD4_classfy.h5')
    model_CD4.build(input_shape)

    ############################################
    ####          New sample test
    data_path = 'E:/cd/Automatic_Gate_Data/test/marker_test/'
    file_list = os.listdir(data_path)

    result_df = pd.DataFrame()

    for info in file_list:
        info_list = list()
        info_list.append(info[:-23])
        # 计算CD3+-的比率
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        ratio_CD3_all, CD3_df = ratioCalculation2(new_df, model_CD3)
        ratio_CD3Pos, ratio_CD3Neg = tuple(ratio_CD3_all)
        info_list.append(ratio_CD3Pos)
        info_list.append(ratio_CD3Neg)
        # 计算CD4+-的比率
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        ratio_CD4_all, CD4_df = ratioCalculation2(new_df, model_CD4)
        ratio_CD4Pos, ratio_CD4Neg = tuple(ratio_CD4_all)
        info_list.append(ratio_CD4Pos)
        info_list.append(ratio_CD4Neg)

        info_df = pd.DataFrame(info_list).T
        info_df.columns = ['id', 'CD3Pos_auto', 'CD3Neg_auto', 'CD4Pos_auto', 'CD4Neg_auto']
        print(info_df)
        result_df = result_df.append(info_df)
        print('Sample %s has finished!' % info[:-23])

    result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)




