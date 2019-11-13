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
    ratio_0 = pre_0_length / length_df * 100
    ratio_1 = pre_1_length / length_df * 100
    ratio_list = [ratio_0, ratio_1]
    return ratio_list



if __name__ == "__main__":

    #### Load Models
    input_shape = (None, 47)

    model_CD66b = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/CD66b_classfy.h5')
    model_CD66b.build(input_shape)


    ############################################
    ####          New sample test
    data_path = 'E:/cd/Automatic_Gate_Data/test/WriteFcs/'
    file_list = os.listdir(data_path)

    result_df = pd.DataFrame()

    for info in file_list:
        new_df = pd.read_csv(data_path+info).iloc[:, :-1]
        ratio_CD66b = ratioCalculation2(new_df, model_CD66b)[:-1]
        ratio_CD66b.insert(0, info[:-18])
        info_df = pd.DataFrame(ratio_CD66b).T
        info_df.columns = ['id', 'CD66b']
        result_df = result_df.append(info_df)
        print('Sample %s has finished!' % info[:-18])

    result_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Output/test.xlsx', index=False)

