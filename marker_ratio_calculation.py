# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/14


import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys, warnings, time, copy

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
    return ratio_list, sub_df, pre_labels


def ratioCalculation3(df, model):
    '''
    计算三分类模型的亚群比率
    :param df:
    :return:
    '''
    test = df.values
    predictions = model.predict(test)
    pre_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
    pre_0_length = len([i for i in pre_labels if i == 0])
    pre_1_length = len([i for i in pre_labels if i == 1])
    pre_2_length = len([i for i in pre_labels if i == 2])
    length_df = df.shape[0]
    df['class'] = pre_labels
    sub_df = df[df['class']==0]
    ratio_0 = pre_0_length / length_df * 100
    ratio_1 = pre_1_length / length_df * 100
    ratio_2 = pre_2_length / length_df * 100
    ratio_list = [ratio_0, ratio_1, ratio_2]
    return ratio_list, sub_df, pre_labels



def markerRatioCalculation(raw_df):
    '''
    计算特定marker的标签矩阵
    :param df: 单个样本的CSV文件
    :return: 特定marker的标签矩阵
    '''
    #### Load Models
    input_shape = (None, 47)
    
    model_Viable = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/Viable_classfy.h5')
    model_Viable.build(input_shape)

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
    
    model_HLA_DR = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/HLA_DR_classfy.h5')
    model_HLA_DR.build(input_shape)
    
    model_CD161 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD161_classfy.h5')
    model_CD161.build(input_shape)

    model_CD56 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD56_classfy.h5')
    model_CD56.build(input_shape)
    
    model_CD197 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD197_classfy.h5')
    model_CD197.build(input_shape)
    
    model_CD68 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD68_classfy.h5')
    model_CD68.build(input_shape)

    model_CD24 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD24_classfy.h5')
    model_CD24.build(input_shape)
    
    model_CD28 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD28_classfy.h5')
    model_CD28.build(input_shape)

    model_ki67 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/ki67_classfy.h5')
    model_ki67.build(input_shape)

    model_PD1 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/PD1_classfy.h5')
    model_PD1.build(input_shape)
    
    model_CD278 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD278_classfy.h5')
    model_CD278.build(input_shape)
    
    model_CCR6 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CCR6_classfy.h5')
    model_CCR6.build(input_shape)   
    
    model_CD123 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD123_classfy.h5')
    model_CD123.build(input_shape)    

    model_CD25 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD25_classfy.h5')
    model_CD25.build(input_shape)

    model_foxp3 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/foxp3_classfy.h5')
    model_foxp3.build(input_shape) 
    
    model_CD274 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD274_classfy.h5')
    model_CD274.build(input_shape)

    model_CD152 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD152_classfy.h5')
    model_CD152.build(input_shape)

    model_CD85j = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD85j_classfy.h5')
    model_CD85j.build(input_shape)

    model_CD183 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD183_classfy.h5')
    model_CD183.build(input_shape)
    
    model_CD38 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD38_classfy.h5')
    model_CD38.build(input_shape)


    # ####          New sample test
    # # data_path = 'E:/cd/Automatic_Gate_Data/test/marker_test/'
    # data_path = 'E:/cd/Automatic_Gate_Data/test/new_test/'
    # file_list = os.listdir(data_path)
    #
    # result_df = pd.DataFrame()

    # for info in file_list:
    # info_list = list()
    # info_list.append(info[:-23])
    label_df = pd.DataFrame()

    start = time.time()

    # raw_df = pd.read_csv(data_path+info).iloc[:, :-1]

    # 计算Viable+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(raw_df)
    ratio_Viable_all, Viable_df, Viable_labels = ratioCalculation2(new_df, model_Viable)
    ratio_ViablePos, ratio_ViableNeg = tuple(ratio_Viable_all)
    # info_list.append(ratio_ViablePos)
    # info_list.append(ratio_ViableNeg)
    label_df = label_df.append(pd.DataFrame(Viable_labels).T)
    Viable_df = Viable_df.iloc[:, :-1]

    # 计算CD3+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD3_all, CD3_df, CD3_labels = ratioCalculation2(new_df, model_CD3)
    ratio_CD3Pos, ratio_CD3Neg = tuple(ratio_CD3_all)
    # info_list.append(ratio_CD3Pos)
    # info_list.append(ratio_CD3Neg)
    label_df = label_df.append(pd.DataFrame(CD3_labels).T)

    # 计算CD4+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD4_all, CD4_df, CD4_labels = ratioCalculation2(new_df, model_CD4)
    ratio_CD4Pos, ratio_CD4Neg = tuple(ratio_CD4_all)
    # info_list.append(ratio_CD4Pos)
    # info_list.append(ratio_CD4Neg)
    label_df = label_df.append(pd.DataFrame(CD4_labels).T)

    # 计算CD8+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD8_all, CD8_df, CD8_labels = ratioCalculation2(new_df, model_CD8)
    ratio_CD8Pos, ratio_CD8Neg = tuple(ratio_CD8_all)
    # info_list.append(ratio_CD8Pos)
    # info_list.append(ratio_CD8Neg)
    label_df = label_df.append(pd.DataFrame(CD8_labels).T)

    # 计算CD45+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD45_all, CD45_df, CD45_labels = ratioCalculation2(new_df, model_CD45)
    ratio_CD45Pos, ratio_CD45Neg = tuple(ratio_CD45_all)
    # info_list.append(ratio_CD45Pos)
    # info_list.append(ratio_CD45Neg)
    label_df = label_df.append(pd.DataFrame(CD45_labels).T)

    # 计算IGD+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_IGD_all, IGD_df, IGD_labels = ratioCalculation2(new_df, model_IGD)
    ratio_IGDPos, ratio_IGDNeg = tuple(ratio_IGD_all)
    # info_list.append(ratio_IGDPos)
    # info_list.append(ratio_IGDNeg)
    label_df = label_df.append(pd.DataFrame(IGD_labels).T)

    # 计算CD11b+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD11b_all, CD11b_df, CD11b_labels = ratioCalculation2(new_df, model_CD11b)
    ratio_CD11bPos, ratio_CD11bNeg = tuple(ratio_CD11b_all)
    # info_list.append(ratio_CD11bPos)
    # info_list.append(ratio_CD11bNeg)
    label_df = label_df.append(pd.DataFrame(CD11b_labels).T)

    # 计算CD14+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD14_all, CD14_df, CD14_labels = ratioCalculation2(new_df, model_CD14)
    ratio_CD14Pos, ratio_CD14Neg = tuple(ratio_CD14_all)
    # info_list.append(ratio_CD14Pos)
    # info_list.append(ratio_CD14Neg)
    label_df = label_df.append(pd.DataFrame(CD14_labels).T)

    # 计算CD19+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD19_all, CD19_df, CD19_labels = ratioCalculation2(new_df, model_CD19)
    ratio_CD19Pos, ratio_CD19Neg = tuple(ratio_CD19_all)
    # info_list.append(ratio_CD19Pos)
    # info_list.append(ratio_CD19Neg)
    label_df = label_df.append(pd.DataFrame(CD19_labels).T)

    # 计算CD20+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD20_all, CD20_df, CD20_labels = ratioCalculation2(new_df, model_CD20)
    ratio_CD20Pos, ratio_CD20Neg = tuple(ratio_CD20_all)
    # info_list.append(ratio_CD20Pos)
    # info_list.append(ratio_CD20Neg)
    label_df = label_df.append(pd.DataFrame(CD20_labels).T)

    # 计算CD27+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD27_all, CD27_df, CD27_labels = ratioCalculation2(new_df, model_CD27)
    ratio_CD27Pos, ratio_CD27Neg = tuple(ratio_CD27_all)
    # info_list.append(ratio_CD27Pos)
    # info_list.append(ratio_CD27Neg)
    label_df = label_df.append(pd.DataFrame(CD27_labels).T)

    # 计算CD33+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD33_all, CD33_df, CD33_labels = ratioCalculation2(new_df, model_CD33)
    ratio_CD33Pos, ratio_CD33Neg = tuple(ratio_CD33_all)
    # info_list.append(ratio_CD33Pos)
    # info_list.append(ratio_CD33Neg)
    label_df = label_df.append(pd.DataFrame(CD33_labels).T)

    # 计算CD39+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD39_all, CD39_df, CD39_labels = ratioCalculation2(new_df, model_CD39)
    ratio_CD39Pos, ratio_CD39Neg = tuple(ratio_CD39_all)
    # info_list.append(ratio_CD39Pos)
    # info_list.append(ratio_CD39Neg)
    label_df = label_df.append(pd.DataFrame(CD39_labels).T)

     # 计算CD86+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD86_all, CD86_df, CD86_labels = ratioCalculation2(new_df, model_CD86)
    ratio_CD86Pos, ratio_CD86Neg = tuple(ratio_CD86_all)
    # info_list.append(ratio_CD86Pos)
    # info_list.append(ratio_CD86Neg)
    label_df = label_df.append(pd.DataFrame(CD86_labels).T)

    # 计算CD94+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD94_all, CD94_df, CD94_labels = ratioCalculation2(new_df, model_CD94)
    ratio_CD94Pos, ratio_CD94Neg = tuple(ratio_CD94_all)
    # info_list.append(ratio_CD94Pos)
    # info_list.append(ratio_CD94Neg)
    label_df = label_df.append(pd.DataFrame(CD94_labels).T)

    # 计算CXCR5+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CXCR5_all, CXCR5_df, CXCR5_labels = ratioCalculation2(new_df, model_CXCR5)
    ratio_CXCR5Pos, ratio_CXCR5Neg = tuple(ratio_CXCR5_all)
    # info_list.append(ratio_CXCR5Pos)
    # info_list.append(ratio_CXCR5Neg)
    label_df = label_df.append(pd.DataFrame(CXCR5_labels).T)

    # 计算gdTCR+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_gdTCR_all, gdTCR_df, gdTCR_labels = ratioCalculation2(new_df, model_gdTCR)
    ratio_gdTCRPos, ratio_gdTCRNeg = tuple(ratio_gdTCR_all)
    # info_list.append(ratio_gdTCRPos)
    # info_list.append(ratio_gdTCRNeg)
    label_df = label_df.append(pd.DataFrame(gdTCR_labels).T)

    # 计算CD57+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD57_all, CD57_df, CD57_labels = ratioCalculation2(new_df, model_CD57)
    ratio_CD57Pos, ratio_CD57Neg = tuple(ratio_CD57_all)
    # info_list.append(ratio_CD57Pos)
    # info_list.append(ratio_CD57Neg)
    label_df = label_df.append(pd.DataFrame(CD57_labels).T)

    # 计算CD11c+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD11c_all, CD11c_df, CD11c_labels = ratioCalculation2(new_df, model_CD11c)
    ratio_CD11cPos, ratio_CD11cNeg = tuple(ratio_CD11c_all)
    # info_list.append(ratio_CD11cPos)
    # info_list.append(ratio_CD11cNeg)
    label_df = label_df.append(pd.DataFrame(CD11c_labels).T)

    # 计算tbet+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_tbet_all, tbet_df, tbet_labels = ratioCalculation2(new_df, model_tbet)
    ratio_tbetPos, ratio_tbetNeg = tuple(ratio_tbet_all)
    # info_list.append(ratio_tbetPos)
    # info_list.append(ratio_tbetNeg)
    label_df = label_df.append(pd.DataFrame(tbet_labels).T)

    # 计算CD16+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD16_all, CD16_df, CD16_labels = ratioCalculation2(new_df, model_CD16)
    ratio_CD16Pos, ratio_CD16Neg = tuple(ratio_CD16_all)
    # info_list.append(ratio_CD16Pos)
    # info_list.append(ratio_CD16Neg)
    label_df = label_df.append(pd.DataFrame(CD16_labels).T)

    # 计算CD127+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD127_all, CD127_df, CD127_labels = ratioCalculation2(new_df, model_CD127)
    ratio_CD127Pos, ratio_CD127Neg = tuple(ratio_CD127_all)
    # info_list.append(ratio_CD127Pos)
    # info_list.append(ratio_CD127Neg)
    label_df = label_df.append(pd.DataFrame(CD127_labels).T)

    # 计算granzyme_B+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_granzyme_B_all, granzyme_B_df, granzyme_B_labels = ratioCalculation2(new_df, model_granzyme_B)
    ratio_granzyme_BPos, ratio_granzyme_BNeg = tuple(ratio_granzyme_B_all)
    # info_list.append(ratio_granzyme_BPos)
    # info_list.append(ratio_granzyme_BNeg)
    label_df = label_df.append(pd.DataFrame(granzyme_B_labels).T)

    # 计算HLA_DR+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_HLA_DR_all, HLA_DR_df, HLA_DR_labels = ratioCalculation2(new_df, model_HLA_DR)
    ratio_HLA_DRPos, ratio_HLA_DRNeg = tuple(ratio_HLA_DR_all)
    # info_list.append(ratio_HLA_DRPos)
    # info_list.append(ratio_HLA_DRNeg)
    label_df = label_df.append(pd.DataFrame(HLA_DR_labels).T)

    # 计算CD161+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD161_all, CD161_df, CD161_labels = ratioCalculation2(new_df, model_CD161)
    ratio_CD161Pos, ratio_CD161Neg = tuple(ratio_CD161_all)
    # info_list.append(ratio_CD161Pos)
    # info_list.append(ratio_CD161Neg)
    label_df = label_df.append(pd.DataFrame(CD161_labels).T)

    # 计算CD56+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD56_all, CD56_df, CD56_labels = ratioCalculation2(new_df, model_CD56)
    ratio_CD56Pos, ratio_CD56Neg = tuple(ratio_CD56_all)
    # info_list.append(ratio_CD56Pos)
    # info_list.append(ratio_CD56Neg)
    label_df = label_df.append(pd.DataFrame(CD56_labels).T)

    # 计算CD197+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD197_all, CD197_df, CD197_labels = ratioCalculation2(new_df, model_CD197)
    ratio_CD197Pos, ratio_CD197Neg = tuple(ratio_CD197_all)
    # info_list.append(ratio_CD197Pos)
    # info_list.append(ratio_CD197Neg)
    label_df = label_df.append(pd.DataFrame(CD197_labels).T)

    # 计算CD68+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD68_all, CD68_df, CD68_labels = ratioCalculation2(new_df, model_CD68)
    ratio_CD68Pos, ratio_CD68Neg = tuple(ratio_CD68_all)
    # info_list.append(ratio_CD68Pos)
    # info_list.append(ratio_CD68Neg)
    label_df = label_df.append(pd.DataFrame(CD68_labels).T)

    # 计算CD24+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD24_all, CD24_df, CD24_labels = ratioCalculation2(new_df, model_CD24)
    ratio_CD24Pos, ratio_CD24Neg = tuple(ratio_CD24_all)
    # info_list.append(ratio_CD24Pos)
    # info_list.append(ratio_CD24Neg)
    label_df = label_df.append(pd.DataFrame(CD24_labels).T)

    # 计算CD28+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD28_all, CD28_df, CD28_labels = ratioCalculation2(new_df, model_CD28)
    ratio_CD28Pos, ratio_CD28Neg = tuple(ratio_CD28_all)
    # info_list.append(ratio_CD28Pos)
    # info_list.append(ratio_CD28Neg)
    label_df = label_df.append(pd.DataFrame(CD28_labels).T)

    # 计算ki67+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_ki67_all, ki67_df, ki67_labels = ratioCalculation2(new_df, model_ki67)
    ratio_ki67Pos, ratio_ki67Neg = tuple(ratio_ki67_all)
    # info_list.append(ratio_ki67Pos)
    # info_list.append(ratio_ki67Neg)
    label_df = label_df.append(pd.DataFrame(ki67_labels).T)

    # 计算PD1+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_PD1_all, PD1_df, PD1_labels = ratioCalculation2(new_df, model_PD1)
    ratio_PD1Pos, ratio_PD1Neg = tuple(ratio_PD1_all)
    # info_list.append(ratio_PD1Pos)
    # info_list.append(ratio_PD1Neg)
    label_df = label_df.append(pd.DataFrame(PD1_labels).T)

    # 计算CD278+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD278_all, CD278_df, CD278_labels = ratioCalculation2(new_df, model_CD278)
    ratio_CD278Pos, ratio_CD278Neg = tuple(ratio_CD278_all)
    # info_list.append(ratio_CD278Pos)
    # info_list.append(ratio_CD278Neg)
    label_df = label_df.append(pd.DataFrame(CD278_labels).T)

    # 计算CCR6+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CCR6_all, CCR6_df, CCR6_labels = ratioCalculation2(new_df, model_CCR6)
    ratio_CCR6Pos, ratio_CCR6Neg = tuple(ratio_CCR6_all)
    # info_list.append(ratio_CCR6Pos)
    # info_list.append(ratio_CCR6Neg)
    label_df = label_df.append(pd.DataFrame(CCR6_labels).T)

    # 计算CD123+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD123_all, CD123_df, CD123_labels = ratioCalculation2(new_df, model_CD123)
    ratio_CD123Pos, ratio_CD123Neg = tuple(ratio_CD123_all)
    # info_list.append(ratio_CD123Pos)
    # info_list.append(ratio_CD123Neg)
    label_df = label_df.append(pd.DataFrame(CD123_labels).T)

    # 计算CD25+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD25_all, CD25_df, CD25_labels = ratioCalculation2(new_df, model_CD25)
    ratio_CD25Pos, ratio_CD25Neg = tuple(ratio_CD25_all)
    # info_list.append(ratio_CD25Pos)
    # info_list.append(ratio_CD25Neg)
    label_df = label_df.append(pd.DataFrame(CD25_labels).T)

    # 计算foxp3+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_foxp3_all, foxp3_df, foxp3_labels = ratioCalculation2(new_df, model_foxp3)
    ratio_foxp3Pos, ratio_foxp3Neg = tuple(ratio_foxp3_all)
    # info_list.append(ratio_foxp3Pos)
    # info_list.append(ratio_foxp3Neg)
    label_df = label_df.append(pd.DataFrame(foxp3_labels).T)

    # 计算CD274+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD274_all, CD274_df, CD274_labels = ratioCalculation2(new_df, model_CD274)
    ratio_CD274Pos, ratio_CD274Neg = tuple(ratio_CD274_all)
    # info_list.append(ratio_CD274Pos)
    # info_list.append(ratio_CD274Neg)
    label_df = label_df.append(pd.DataFrame(CD274_labels).T)

    # 计算CD152+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD152_all, CD152_df, CD152_labels = ratioCalculation2(new_df, model_CD152)
    ratio_CD152Pos, ratio_CD152Neg = tuple(ratio_CD152_all)
    # info_list.append(ratio_CD152Pos)
    # info_list.append(ratio_CD152Neg)
    label_df = label_df.append(pd.DataFrame(CD152_labels).T)

    # 计算CD85j+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD85j_all, CD85j_df, CD85j_labels = ratioCalculation2(new_df, model_CD85j)
    ratio_CD85jPos, ratio_CD85jNeg = tuple(ratio_CD85j_all)
    # info_list.append(ratio_CD85jPos)
    # info_list.append(ratio_CD85jNeg)
    label_df = label_df.append(pd.DataFrame(CD85j_labels).T)

    # 计算CD183+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD183_all, CD183_df, CD183_labels = ratioCalculation2(new_df, model_CD183)
    ratio_CD183Pos, ratio_CD183Neg = tuple(ratio_CD183_all)
    # info_list.append(ratio_CD183Pos)
    # info_list.append(ratio_CD183Neg)
    label_df = label_df.append(pd.DataFrame(CD183_labels).T)

    # 计算CD38+-的比率
    # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    new_df = copy.deepcopy(Viable_df)
    ratio_CD38_all, CD38_df, CD38_labels = ratioCalculation2(new_df, model_CD38)
    ratio_CD38Pos, ratio_CD38Neg = tuple(ratio_CD38_all)
    # info_list.append(ratio_CD38Pos)
    # info_list.append(ratio_CD38Neg)
    label_df = label_df.append(pd.DataFrame(CD38_labels).T)

    # # 计算CD38_0_1_2的比率
    # # new_df = pd.read_csv(data_path+info).iloc[:, :-1]
    # new_df = copy.deepcopy(Viable_df)
    # ratio_CD38_all, CD38_df, CD38_labels = ratioCalculation3(new_df, model_CD38)
    # ratio_CD38_0, ratio_CD38_1, ratio_CD38_2 = tuple(ratio_CD38_all)
    # info_list.append(ratio_CD38_0)
    # info_list.append(ratio_CD38_1)
    # info_list.append(ratio_CD38_2)
    # label_df = label_df.append(pd.DataFrame(CD38_labels).T)

    print('Label prediction has finished!', '\n', '\n')
    # print('Now start to write out the data. This process is time consuming. Please be patient.^_^')

    # 细胞类标
    label_df = label_df.T
    label_df.columns = ['Viable',
                        'CD3',
                        'CD4',
                        'CD8',
                        'CD45',
                        'IGD',
                        'CD11b',
                        'CD14',
                        'CD19',
                        'CD20',
                        'CD27',
                        'CD33',
                        'CD39',
                        'CD86',
                        'CD94',
                        'CXCR5',
                        'gdTCR',
                        'CD57',
                        'CD11c',
                        'tbet',
                        'CD16',
                        'CD127',
                        'granzyme_B',
                        'HLA_DR',
                        'CD161',
                        'CD56',
                        'CD197',
                        'CD68',
                        'CD24',
                        'CD28',
                        'ki67',
                        'PD1',
                        'CD278',
                        'CCR6',
                        'CD123',
                        'CD25',
                        'FOXP3',
                        'CD274',
                        'CD152',
                        'CD85j',
                        'CD183',
                        'CD38',
                        ]
    print('Time cost is %s.' % (time.time()-start))
    print('-'*100)
    return label_df
        # label_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/label_prediction/test/%s.xlsx' % info[:4], index=False)


    #     # marker自动圈门得出的比率
    #     info_df = pd.DataFrame(info_list).T
    #     info_df.columns = ['id',
    #                        'CD3Pos_auto', 'CD3Neg_auto',
    #                        'CD4Pos_auto', 'CD4Neg_auto',
    #                        'CD8Pos_auto', 'CD8Neg_auto',
    #                        'CD45Pos_auto', 'CD45Neg_auto',
    #                        'IGDPos_auto', 'IGDNeg_auto',
    #                        'CD11bPos_auto', 'CD11bNeg_auto',
    #                        'CD14Pos_auto', 'CD14Neg_auto',
    #                        'CD19Pos_auto', 'CD19Neg_auto',
    #                        'CD20Pos_auto', 'CD20Neg_auto',
    #                        'CD27Pos_auto', 'CD27Neg_auto',
    #                        'CD33Pos_auto', 'CD33Neg_auto',
    #                        'CD39Pos_auto', 'CD39Neg_auto',
    #                        'CD86Pos_auto', 'CD86Neg_auto',
    #                        'CD94Pos_auto', 'CD94Neg_auto',
    #                        'CXCR5Pos_auto', 'CXCR5Neg_auto',
    #                        'gdTCRPos_auto', 'gdTCRNeg_auto',
    #                        'CD57Pos_auto', 'CD57Neg_auto',
    #                        'CD11cPos_auto', 'CD11cNeg_auto',
    #                        'tbetPos_auto', 'tbetNeg_auto',
    #                        'CD16Pos_auto', 'CD16Neg_auto',
    #                        'CD127Pos_auto', 'CD127Neg_auto',
    #                        'granzyme_BPos_auto', 'granzyme_BNeg_auto',
    #                        'HLA_DRPos_auto', 'HLA_DRNeg_auto',
    #                        'CD161Pos_auto', 'CD161Neg_auto',
    #                        'CD56Pos_auto', 'CD56Neg_auto',
    #                        'CD197Pos_auto', 'CD197Neg_auto',
    #                        'CD68Pos_auto', 'CD68Neg_auto',
    #                        'CD24Pos_auto', 'CD24Neg_auto',
    #                        'CD28Pos_auto', 'CD28Neg_auto',
    #                        'ki67Pos_auto', 'ki67Neg_auto',
    #                        'PD1Pos_auto', 'PD1Neg_auto',
    #                        'CD278Pos_auto', 'CD278Neg_auto',
    #                        'CCR6Pos_auto', 'CCR6Neg_auto',
    #                        'CD123Pos_auto', 'CD123Neg_auto',
    #                        'CD25Pos_auto', 'CD25Neg_auto',
    #                        'foxp3Pos_auto', 'foxp3Neg_auto',
    #                        'CD274Pos_auto', 'CD274Neg_auto',
    #                        'CD152Pos_auto', 'CD152Neg_auto',
    #                        'CD85jPos_auto', 'CD85jNeg_auto',
    #                        'CD183Pos_auto', 'CD183Neg_auto',
    #                        'CD38Pos_auto', 'CD38Neg_auto',
    #                        # 'CD38_0_auto', 'CD38_1_auto', 'CD38_2_auto',
    #                        ]
    #     print(info_df)
    #     result_df = result_df.append(info_df)
    #     print('Sample %s has finished!' % info[:4])
    #     print('Time cost is %s.' % (time.time()-start))
    #     print('-'*100)
    #
    # result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)