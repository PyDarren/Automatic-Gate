# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/8


import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys, warnings

warnings.filterwarnings(action='ignore')


def ratioPrediction(df):
    '''
    Calculate the specific ratio of immune cell subsets
    :param df: rawdata
    :return:
    '''
    #### Calculating the ratio of 27_lymphocytes__0 and 29_monocytes__1
    df_27_29 = df
    test_27_29 = df_27_29.values
    predictions_27_29 = model_27_29.predict(test_27_29)
    pre_labels_27_29 = [np.argmax(predictions_27_29[i]) for i in range(predictions_27_29.shape[0])]
    pre_27_length = len([i for i in pre_labels_27_29 if i == 0])
    pre_29_length = len([i for i in pre_labels_27_29 if i == 1])
    length_27_29 = df_27_29.shape[0]
    df_27_29['class'] = pre_labels_27_29
    subset_27_ratio = pre_27_length / length_27_29 * 100
    subset_29_ratio = pre_29_length / length_27_29 * 100
    print('亚群lymphocytes的比率为：%s' % subset_27_ratio)
    print('亚群monocytes的比率为：%s' % subset_29_ratio)

    #### Calculating the ratio of 18_T cells __0 and 35_CD3-__1
    df_18_35 = df_27_29[df_27_29['class'] == 0]
    del df_18_35['class']
    test_18_35 = df_18_35.values
    predictions_18_35 = model_18_35.predict(test_18_35)
    pre_labels_18_35 = [np.argmax(predictions_18_35[i]) for i in range(predictions_18_35.shape[0])]
    pre_18_length = len([i for i in pre_labels_18_35 if i == 0])
    pre_35_length = len([i for i in pre_labels_18_35 if i == 1])
    length_18_35 = df_18_35.shape[0]
    df_18_35['class'] = pre_labels_18_35
    subset_18_ratio = pre_18_length / length_18_35 * 100
    subset_35_ratio = pre_35_length / length_18_35 * 100
    print('亚群T cells的比率为：%s' % subset_18_ratio)
    print('亚群CD3-的比率为：%s' % subset_35_ratio)

    #### Calculating the ratio of 26_gamma-delta T cells
    df_26_no26 = df_27_29[df_27_29['class'] == 0]
    del df_26_no26['class']
    test_26_no26 = df_26_no26.values
    predictions_26_no26 = model_26_no26.predict(test_26_no26)
    pre_labels_26_no26 = [np.argmax(predictions_26_no26[i]) for i in range(predictions_26_no26.shape[0])]
    pre_26_length = len([i for i in pre_labels_26_no26 if i == 0])
    pre_no26_length = len([i for i in pre_labels_26_no26 if i == 1])
    length_26_no26 = df_26_no26.shape[0]
    df_26_no26['class'] = pre_labels_26_no26
    subset_26_ratio = pre_26_length / length_26_no26 * 100
    subset_no26_ratio = pre_no26_length / length_26_no26 * 100
    print('亚群gamma delta T cells的比率为：%s' % subset_26_ratio)

    #### Calculating the ratio of 17 NK T cells
    df_17_no17 = df_27_29[df_27_29['class'] == 0]
    del df_17_no17['class']
    test_17_no17 = df_17_no17.values
    predictions_17_no17 = model_17_no17.predict(test_17_no17)
    pre_labels_17_no17 = [np.argmax(predictions_17_no17[i]) for i in range(predictions_17_no17.shape[0])]
    pre_17_length = len([i for i in pre_labels_17_no17 if i == 0])
    pre_no17_length = len([i for i in pre_labels_17_no17 if i == 1])
    length_17_no17 = df_17_no17.shape[0]
    df_17_no17['class'] = pre_labels_17_no17
    subset_17_ratio = pre_17_length / length_17_no17 * 100
    subset_no17_ratio = pre_no17_length / length_17_no17 * 100
    print('亚群NK T cells的比率为：%s' % subset_17_ratio)

    #### Calculating the ratio of 8_CD4+ T cell__0 and 10_CD8+ T cell__1 and others__2
    df_8_10_other = df_18_35[df_18_35['class'] == 0]
    del df_8_10_other['class']
    test_8_10_other = df_8_10_other.values
    predictions_8_10_other = model_8_10_other.predict(test_8_10_other)
    pre_labels_8_10_other = [np.argmax(predictions_8_10_other[i]) for i in range(predictions_8_10_other.shape[0])]
    pre_8_length = len([i for i in pre_labels_8_10_other if i == 0])
    pre_10_length = len([i for i in pre_labels_8_10_other if i == 1])
    pre_other_length = len([i for i in pre_labels_8_10_other if i == 2])
    length_8_10_other = df_8_10_other.shape[0]
    df_8_10_other['class'] = pre_labels_8_10_other
    subset_8_ratio = pre_8_length / length_8_10_other * 100
    subset_10_ratio = pre_10_length / length_8_10_other * 100
    print('亚群CD4+ T Cells的比率为：%s' % subset_8_ratio)
    print('亚群CD8+ T Cells的比率为：%s' % subset_10_ratio)

    ratio_list = [subset_27_ratio, subset_29_ratio, subset_18_ratio, subset_26_ratio, subset_17_ratio, subset_8_ratio, subset_10_ratio]
    return ratio_list



if __name__ == '__main__':

    #### Load Models
    input_shape = (None, 42)

    model_27_29 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/27_29_classfy.h5')
    model_27_29.build(input_shape)

    model_18_35 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/18_35_classfy.h5')
    model_18_35.build(input_shape)

    model_26_no26 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/26_no26_classfy.h5')
    model_26_no26.build(input_shape)

    model_17_no17 = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/17_no17_classfy.h5')
    model_17_no17.build(input_shape)
    
    model_8_10_other = tf.keras.models.load_model('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/8_10_other_classfy.h5')
    model_8_10_other.build(input_shape)
    
    
    ############################################
    ####          New sample test

    data_path = 'E:/cd/Automatic_Gate_Data/Export/v/Marker_rename/Stain_excel/'
    file_list = os.listdir(data_path)

    result_df = pd.DataFrame()

    for info in file_list:
        new_df = pd.read_csv(data_path+info)
        ratio_list = ratioPrediction(new_df)
        ratio_list.insert(0, info[:-4])
        info_df = pd.DataFrame(ratio_list).T
        info_df.columns = ['id', 'lymphocytes_Auto', 'monocytes_Auto', 'T cells_Auto', 'gamma delta T cells_Auto', 'NK T cells_Auto',
                           'CD4+ T cells_Auto', 'CD8+ T cells_Auto']
        result_df = result_df.append(info_df)

    result_df.to_excel('E:/cd/Automatic_Gate_Data/test/autogate_result.xlsx', index=False)
