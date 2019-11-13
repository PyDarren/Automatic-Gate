# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/8


import pandas as pd
import numpy as np
import random



def sample_func(df, num):
    df_index = list(df.index)
    select_index = random.sample(df_index, num)
    new_df = df.iloc[select_index, :]
    print(new_df.shape)
    return new_df



def split_func(data_frame, size=0.9):
    """
    Split the data into two data set
    :param data_frame: the name of input data
    :param size : the size of train data
    :return: train_data, test_data
    """
    data_frame = pd.DataFrame(data_frame.values,
                              index=[i for i in range(data_frame.values.shape[0])],
                              columns=data_frame.columns)

    healthy_data = data_frame[data_frame["class"] == 0]
    unhealthy_data = data_frame[data_frame["class"] == 1]

    healthy_index = list(healthy_data.index)
    unhealthy_index = list(unhealthy_data.index)

    healthy_train_data_index = random.sample(healthy_index, int(size * len(healthy_index)))
    unhealthy_train_data_index = random.sample(unhealthy_index, int(size * len(unhealthy_index)))

    healthy_test_data_index = list(set(healthy_index).difference(set(healthy_train_data_index)))
    unhealthy_test_data_index = list(set(unhealthy_index).difference(set(unhealthy_train_data_index)))

    train_index = list(set(healthy_train_data_index).union(set(unhealthy_train_data_index)))
    test_index = list(set(healthy_test_data_index).union(set(unhealthy_test_data_index)))

    train = data_frame.iloc[train_index, :]
    test = data_frame.iloc[test_index, :]

    return train, test




if __name__ == '__main__':

    #### Data import
    data_path = 'E:/cd/Automatic_Gate_Data/Rawdata/'
    file_0 = '17_NKT.csv'
    file_1 = 'no17.csv'
    pair_name = '17_no17'
    df_0 = pd.read_csv(data_path+file_0).iloc[:, 1:]
    df_1 = pd.read_csv(data_path+file_1).iloc[:, 1:]

    #### Add category variable
    df_0['class'] = 0
    df_1['class'] = 1

    #### Random select some data
    # new_df_0 = sample_func(df_0, 500000)
    # new_df_1 = sample_func(df_1, 500000)
    new_df_0 = df_0
    new_df_1 = sample_func(df_1, len(new_df_0))

    #### Merge all data
    final_df = new_df_0.append(new_df_1)
    final_df.index = [i for i in range(final_df.shape[0])]
    print('Finish merge.')

    #### Divide the training set and the test set
    train, test = split_func(final_df)
    print('Finish divide.')

    #### Export data
    train.to_csv(data_path + 'traindata_%s.csv' % pair_name, index=False)
    test.to_csv(data_path + 'testdata_%s.csv' % pair_name, index=False)
    final_df.to_csv(data_path + 'rawdata_%s.csv' % pair_name, index=False)
