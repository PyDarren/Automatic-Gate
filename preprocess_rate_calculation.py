# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/12



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

    model_CD66b = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD66b_classfy.h5')
    model_CD66b.build(input_shape)

    model_beads = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/beads_classfy.h5')
    model_beads.build(input_shape)

    model_DNA = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/DNA_classfy.h5')
    model_DNA.build(input_shape)


    ############################################
    ####          New sample test
    data_path = 'E:/cd/Automatic_Gate_Data/test/WriteFcs/'
    file_list = os.listdir(data_path)

    result_df = pd.DataFrame()

    for info in file_list:
        info_list = list()
        info_list.append(info[:-18])
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # 计算CD66b_比率
        ratio_CD66b_all, CD66b_df = ratioCalculation2(new_df, model_CD66b)
        ratio_CD66bNeg = ratio_CD66b_all[0]
        info_list.append(ratio_CD66bNeg)
        # 计算remove beads比率
        CD66b_df_new = CD66b_df.iloc[:, :-1]
        ratio_beads_all, beads_df = ratioCalculation2(CD66b_df_new, model_beads)
        ratio_remove_beads = ratio_beads_all[0]
        info_list.append(ratio_remove_beads)
        # 计算DNA比率
        beads_df_new = beads_df.iloc[:, :-1]
        ratio_DNA_all, DNA_df = ratioCalculation2(beads_df_new, model_DNA)
        ratio_DNA = ratio_DNA_all[0]
        info_list.append(ratio_DNA)
        # 计算singlets比率
        DNA_length = len(DNA_df)
        singlets_length = len(DNA_df[DNA_df['length'] <= 20])
        ratio_singlets = singlets_length / DNA_length
        info_list.append(ratio_singlets)

        info_df = pd.DataFrame(info_list).T
        info_df.columns = ['id', 'CD66b_auto', 'Remove beads_auto', 'DNA_auto', 'Singlets_auto']
        print(info_df)
        result_df = result_df.append(info_df)
        print('Sample %s has finished!' % info[:-18])

    result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)

