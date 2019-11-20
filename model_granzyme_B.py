# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/20



import pandas as pd
import numpy as np
import tensorflow as tf
import random



def sample_func(df, num):
    df_index = list(df.index)
    select_index = random.sample(df_index, num)
    new_df = df.iloc[select_index, :]
    print(new_df.shape)
    return new_df



def balance_train(data_frame, more_label=0, less_label=1):
    '''
    Balance number of positive and negative cases in data_frame
    # 欠采样
    :param data_frame:
    :return:
    '''
    more_class = data_frame[data_frame['class'] == more_label]
    less_class  = data_frame[data_frame['class'] == less_label]
    num = len(less_class)
    more_index = list(more_class.index)
    choose_list = list()
    for i in range(num):
        index_choose = random.sample(more_index, 1)[0]
        choose_list.append(index_choose)
    more_choose = more_class.loc[choose_list, :]
    data_new = more_choose.append(less_class)
    data_new.index = [i for i in range(data_new.shape[0])]
    return data_new



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



if __name__ == "__main__":

    ######################################
    #### Data import
    data_path = 'E:/cd/Automatic_Gate_Data/Rawdata/marker_42/'
    file_0 = 'Granzyme_B+'
    file_1 = 'Granzyme_B-'
    df_0 = pd.read_csv(data_path+file_0+'.csv').iloc[:, :-1]
    df_1 = pd.read_csv(data_path+file_1+'.csv').iloc[:, :-1]

    #### Add category variable
    df_0['class'] = 0
    df_1['class'] = 1

    # df_0 = sample_func(df_0, 500000)
    # df_1 = sample_func(df_1, 500000)

    #### Merge all data
    final_df = df_0.append(df_1)
    final_df.index = [i for i in range(final_df.shape[0])]
    print('Finish merge.')

    #### Divide the training set and the test set
    train_raw, test = split_func(final_df)
    train = balance_train(train_raw, more_label=1, less_label=0)
    print('Finish divide.')


    ##############################################
    ####           Model
    train_X = train.iloc[:, :-1].values
    train_labels = train['class'].values
    test_X = test.iloc[:, :-1].values
    test_labels = test['class'].values
    test_less_X = test[test['class'] == 1].iloc[:, :-1].values
    test_less_labels = test[test['class'] == 1]['class'].values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(64, activation='tanh'),
        # tf.keras.layers.Dense(64, activation='tanh'),
        # tf.keras.layers.Dense(32, activation='tanh'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])


    optimizer = tf.keras.optimizers.Adam(0.0005)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X,
              train_labels,
              epochs=10,
              # batch_size=16384,
              # validation_data=(test_X, test_labels),
              # verbose=2
              )

    test_loss, test_acc =  model.evaluate(test_X, test_labels,verbose=2)
    print('\nTest accuracy:', test_acc)
    test_less_loss, test_less_acc =  model.evaluate(test_less_X, test_less_labels,verbose=2)
    print('\nTest less accuracy:', test_less_acc)

    ## save model
    model.save('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/granzyme_B_classfy.h5')


