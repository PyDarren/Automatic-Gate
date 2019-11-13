# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/8


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def oneHotEncode(df):
    df['label_27'] = np.where(df['class']==1, 1, 0)
    df['label_29'] = np.where(df['class']==2, 1, 0)
    del df['class']
    return df


def toDataset(df, batch_size):
    df_X = df.iloc[:, :-2].values
    df_y = df.iloc[:, -2:].values
    data = tf.data.Dataset.from_tensor_slices((df_X, df_y))
    data = data.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(data.output_types,
                                               data.output_shapes)
    init_op = iterator.make_initializer(data)
    return iterator, init_op




if __name__ == '__main__':

    ######################################################################
    ##                            Data Import
    data_path = 'E:/cd/Automatic_Gate_Data/Rawdata/'
    train_data = pd.read_csv(data_path+'traindata_8_10_other.csv')
    test_data = pd.read_csv(data_path+'testdata_8_10_other.csv')


    ######################################################################
    ##                                Model

    # 定义参数
    learning_rate = 0.03
    batch_size = 1000
    training_enpochs = int(len(train_data)/batch_size)
    display_step = 1

    train_X = train_data.iloc[:, :-1].values
    train_labels = train_data['class'].values
    test_X = test_data.iloc[:, :-1].values
    test_labels = test_data['class'].values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(42, activation='sigmoid'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X, train_labels, epochs=10)

    test_loss, test_acc =  model.evaluate(test_X, test_labels,verbose=2)
    print('\nTest accuracy:', test_acc)

    ## save model
    model.save('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/8_10_other_classfy.h5')



