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

    model_CD33 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD33_classfy.h5')
    model_CD33.build(input_shape)

    model_CD39 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD39_classfy.h5')
    model_CD39.build(input_shape)

    model_CD86 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD86_classfy.h5')
    model_CD86.build(input_shape)
    
    model_CD94 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD94_classfy.h5')
    model_CD94.build(input_shape)
    
    model_CXCR5 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CXCR5_classfy.h5')
    model_CXCR5.build(input_shape)

    model_gdTCR = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/gdTCR_classfy.h5')
    model_gdTCR.build(input_shape)

    model_CD57 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD57_classfy.h5')
    model_CD57.build(input_shape)
    
    model_CD11c = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD11c_classfy.h5')
    model_CD11c.build(input_shape)
    
    model_tbet = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/tbet_classfy.h5')
    model_tbet.build(input_shape)

    model_CD16 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD16_classfy.h5')
    model_CD16.build(input_shape)

    model_CD127 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD127_classfy.h5')
    model_CD127.build(input_shape)
    
    model_granzyme_B = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/granzyme_B_classfy.h5')
    model_granzyme_B.build(input_shape)



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
        
        # # 计算CD27+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD27_all, CD27_df = ratioCalculation2(new_df, model_CD27)
        # ratio_CD27Pos, ratio_CD27Neg = tuple(ratio_CD27_all)
        # info_list.append(ratio_CD27Pos)
        # info_list.append(ratio_CD27Neg)

        # # 计算CD33+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD33_all, CD33_df = ratioCalculation2(new_df, model_CD33)
        # ratio_CD33Pos, ratio_CD33Neg = tuple(ratio_CD33_all)
        # info_list.append(ratio_CD33Pos)
        # info_list.append(ratio_CD33Neg)

        # # 计算CD39+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD39_all, CD39_df = ratioCalculation2(new_df, model_CD39)
        # ratio_CD39Pos, ratio_CD39Neg = tuple(ratio_CD39_all)
        # info_list.append(ratio_CD39Pos)
        # info_list.append(ratio_CD39Neg)
        
        #  # 计算CD86+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD86_all, CD86_df = ratioCalculation2(new_df, model_CD86)
        # ratio_CD86Pos, ratio_CD86Neg = tuple(ratio_CD86_all)
        # info_list.append(ratio_CD86Pos)
        # info_list.append(ratio_CD86Neg)
        
        # # 计算CD94+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD94_all, CD94_df = ratioCalculation2(new_df, model_CD94)
        # ratio_CD94Pos, ratio_CD94Neg = tuple(ratio_CD94_all)
        # info_list.append(ratio_CD94Pos)
        # info_list.append(ratio_CD94Neg)

        # # 计算CXCR5+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CXCR5_all, CXCR5_df = ratioCalculation2(new_df, model_CXCR5)
        # ratio_CXCR5Pos, ratio_CXCR5Neg = tuple(ratio_CXCR5_all)
        # info_list.append(ratio_CXCR5Pos)
        # info_list.append(ratio_CXCR5Neg)
        
        # # 计算gdTCR+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_gdTCR_all, gdTCR_df = ratioCalculation2(new_df, model_gdTCR)
        # ratio_gdTCRPos, ratio_gdTCRNeg = tuple(ratio_gdTCR_all)
        # info_list.append(ratio_gdTCRPos)
        # info_list.append(ratio_gdTCRNeg)

        # # 计算CD57+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD57_all, CD57_df = ratioCalculation2(new_df, model_CD57)
        # ratio_CD57Pos, ratio_CD57Neg = tuple(ratio_CD57_all)
        # info_list.append(ratio_CD57Pos)
        # info_list.append(ratio_CD57Neg)

        # # 计算CD11c+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD11c_all, CD11c_df = ratioCalculation2(new_df, model_CD11c)
        # ratio_CD11cPos, ratio_CD11cNeg = tuple(ratio_CD11c_all)
        # info_list.append(ratio_CD11cPos)
        # info_list.append(ratio_CD11cNeg)

        # # 计算tbet+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_tbet_all, tbet_df = ratioCalculation2(new_df, model_tbet)
        # ratio_tbetPos, ratio_tbetNeg = tuple(ratio_tbet_all)
        # info_list.append(ratio_tbetPos)
        # info_list.append(ratio_tbetNeg)
        
        # # 计算CD16+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD16_all, CD16_df = ratioCalculation2(new_df, model_CD16)
        # ratio_CD16Pos, ratio_CD16Neg = tuple(ratio_CD16_all)
        # info_list.append(ratio_CD16Pos)
        # info_list.append(ratio_CD16Neg)

        # # 计算CD127+-的比率
        # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        # ratio_CD127_all, CD127_df = ratioCalculation2(new_df, model_CD127)
        # ratio_CD127Pos, ratio_CD127Neg = tuple(ratio_CD127_all)
        # info_list.append(ratio_CD127Pos)
        # info_list.append(ratio_CD127Neg)
        
        # 计算granzyme_B+-的比率
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        ratio_granzyme_B_all, granzyme_B_df = ratioCalculation2(new_df, model_granzyme_B)
        ratio_granzyme_BPos, ratio_granzyme_BNeg = tuple(ratio_granzyme_B_all)
        info_list.append(ratio_granzyme_BPos)
        info_list.append(ratio_granzyme_BNeg)        
        

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
                           # 'CD27Pos_auto', 'CD27Neg_auto',
                           # 'CD33Pos_auto', 'CD33Neg_auto',
                           # 'CD39Pos_auto', 'CD39Neg_auto',
                           # 'CD86Pos_auto', 'CD86Neg_auto',
                           # 'CD94Pos_auto', 'CD94Neg_auto',
                           # 'CXCR5Pos_auto', 'CXCR5Neg_auto',
                           # 'gdTCRPos_auto', 'gdTCRNeg_auto',
                           # 'CD57Pos_auto', 'CD57Neg_auto',
                           # 'CD11cPos_auto', 'CD11cNeg_auto',
                           # 'tbetPos_auto', 'tbetNeg_auto',
                           # 'CD16Pos_auto', 'CD16Neg_auto',
                           # 'CD127Pos_auto', 'CD127Neg_auto',
                           'granzyme_BPos_auto', 'granzyme_BNeg_auto',
                           ]
        print(info_df)
        result_df = result_df.append(info_df)
        print('Sample %s has finished!' % info[:-23])
        print('-'*100)

    result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)