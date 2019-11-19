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

    model_CD8 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD8_classfy.h5')
    model_CD8.build(input_shape)
    
    model_CD45 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD45_classfy.h5')
    model_CD45.build(input_shape)
    
    model_IGD = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/IGD_classfy.h5')
    model_IGD.build(input_shape)

    model_CD11b = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD11b_classfy.h5')
    model_CD11b.build(input_shape)

    model_CD14 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD14_classfy.h5')
    model_CD14.build(input_shape)
    
    model_CD19 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD19_classfy.h5')
    model_CD19.build(input_shape)

    model_CD20 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD20_classfy.h5')
    model_CD20.build(input_shape)
    
    model_CD27 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD27_classfy.h5')
    model_CD27.build(input_shape)


    ############################################
    ####          New sample test
    data_path = 'E:/cd/Automatic_Gate_Data/test/marker_test/'
    file_list = os.listdir(data_path)

    result_df = pd.DataFrame()

    for info in file_list:
        info_list = list()
        info_list.append(info[:-23])

        # # 计算CD3+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD3_all, CD3_df = ratioCalculation2(new_df, model_CD3)
        # ratio_CD3Pos, ratio_CD3Neg = tuple(ratio_CD3_all)
        # info_list.append(ratio_CD3Pos)
        # info_list.append(ratio_CD3Neg)
        #
        # # 计算CD4+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD4_all, CD4_df = ratioCalculation2(new_df, model_CD4)
        # ratio_CD4Pos, ratio_CD4Neg = tuple(ratio_CD4_all)
        # info_list.append(ratio_CD4Pos)
        # info_list.append(ratio_CD4Neg)
        #
        # # 计算CD8+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD8_all, CD8_df = ratioCalculation2(new_df, model_CD8)
        # ratio_CD8Pos, ratio_CD8Neg = tuple(ratio_CD8_all)
        # info_list.append(ratio_CD8Pos)
        # info_list.append(ratio_CD8Neg)
        #
        # # 计算CD45+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD45_all, CD45_df = ratioCalculation2(new_df, model_CD45)
        # ratio_CD45Pos, ratio_CD45Neg = tuple(ratio_CD45_all)
        # info_list.append(ratio_CD45Pos)
        # info_list.append(ratio_CD45Neg)
        #
        # # 计算IGD+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_IGD_all, IGD_df = ratioCalculation2(new_df, model_IGD)
        # ratio_IGDPos, ratio_IGDNeg = tuple(ratio_IGD_all)
        # info_list.append(ratio_IGDPos)
        # info_list.append(ratio_IGDNeg)
        #
        # # 计算CD11b+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD11b_all, CD11b_df = ratioCalculation2(new_df, model_CD11b)
        # ratio_CD11bPos, ratio_CD11bNeg = tuple(ratio_CD11b_all)
        # info_list.append(ratio_CD11bPos)
        # info_list.append(ratio_CD11bNeg)

        # # 计算CD14+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD14_all, CD14_df = ratioCalculation2(new_df, model_CD14)
        # ratio_CD14Pos, ratio_CD14Neg = tuple(ratio_CD14_all)
        # info_list.append(ratio_CD14Pos)
        # info_list.append(ratio_CD14Neg)
        
        # # 计算CD19+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD19_all, CD19_df = ratioCalculation2(new_df, model_CD19)
        # ratio_CD19Pos, ratio_CD19Neg = tuple(ratio_CD19_all)
        # info_list.append(ratio_CD19Pos)
        # info_list.append(ratio_CD19Neg)

        # # 计算CD20+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD20_all, CD20_df = ratioCalculation2(new_df, model_CD20)
        # ratio_CD20Pos, ratio_CD20Neg = tuple(ratio_CD20_all)
        # info_list.append(ratio_CD20Pos)
        # info_list.append(ratio_CD20Neg)
        
        # 计算CD27+-的比率
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        ratio_CD27_all, CD27_df = ratioCalculation2(new_df, model_CD27)
        ratio_CD27Pos, ratio_CD27Neg = tuple(ratio_CD27_all)
        info_list.append(ratio_CD27Pos)
        info_list.append(ratio_CD27Neg)        

        info_df = pd.DataFrame(info_list).T
        info_df.columns = ['id',
                           # 'CD3Pos_auto', 'CD3Neg_auto',
                           # 'CD4Pos_auto', 'CD4Neg_auto',
                           # 'CD8Pos_auto', 'CD8Neg_auto',
                           # 'CD45Pos_auto', 'CD45Neg_auto',
                           # 'IGDPos_auto', 'IGDNeg_auto',
                           # 'CD11bPos_auto', 'CD11bNeg_auto',
                           # 'CD14Pos_auto', 'CD14Neg_auto',
                           # 'CD19Pos_auto', 'CD19Neg_auto',
                           # 'CD20Pos_auto', 'CD20Neg_auto',
                           'CD27Pos_auto', 'CD27Neg_auto']
        print(info_df)
        result_df = result_df.append(info_df)
        print('Sample %s has finished!' % info[:-23])
        print('-'*100)

    result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)
