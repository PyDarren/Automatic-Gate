# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/11/6


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
    train_data = pd.read_csv(data_path+'traindata_27_29.csv')
    test_data = pd.read_csv(data_path+'testdata_27_29.csv')

    # train_data = oneHotEncode(train_data)
    # test_data = oneHotEncode(test_data)



    ######################################################################
    ##                                Model

    # 定义参数
    learning_rate = 0.03
    batch_size = 1000
    training_enpochs = int(len(train_data)/batch_size)
    display_step = 1

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data.iloc[:, :-1].values, train_data.iloc[:, -1:]))
    # train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size)
    #
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_data.iloc[:, :-1].values, test_data.iloc[:, -1:]))
    # test_dataset = test_dataset.batch(batch_size)

    train_X = train_data.iloc[:, :-1].values
    train_labels = train_data['class'].values
    test_X = test_data.iloc[:, :-1].values
    test_labels = test_data['class'].values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(42, activation='sigmoid'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_X, train_labels, epochs=100)

    test_loss, test_acc =  model.evaluate(test_X, test_labels,verbose=2)
    print('\nTest accuracy:', test_acc)

    ## save model
    model.save('C:/Users/pc/OneDrive/PLTTECH/Project/01_自动圈门建模/Models/27_29_classfy.h5')



    ## new sample test
    new_df = pd.read_csv('E:/cd/Automatic_Gate_Data/test/v/B003/Marker_rename/Stain_excel/B003.csv').iloc[:, 1:]
    new_test = new_df.values
    predictions = model.predict(new_test)
    pre_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
    pre_0_length = len([i for i in pre_labels if i == 0])
    pre_1_length = len([i for i in pre_labels if i == 1])
    sample_length = new_df.shape[0]
    subset_27_ratio = pre_0_length / sample_length
    subset_29_ratio = pre_1_length / sample_length
    print('亚群lymphocytes的比率为：%s' % subset_27_ratio)
    print('亚群monocytes的比率为：%s' % subset_29_ratio)




    #
    # # 定义占位符
    # x = tf.placeholder(tf.float32, [None, 42])
    # y = tf.placeholder(tf.float32, [None, 2])
    #
    # # 定义学习参数
    # W = tf.Variable(tf.random_normal([42, 2]))
    # b = tf.Variable(tf.zeros([2]))
    #
    # # 定义输出节点
    # pred = tf.nn.softmax(tf.matmul(x, W) + b)
    #
    # # 定义损失函数
    # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    #
    # # 梯度下降优化器
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    #
    # # 转换数据成dataset类
    # iterator_train, init_op_train = toDataset(train_data, batch_size)
    #
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     # 启动循环开始训练
    #     for epoch in range(training_enpochs):
    #         avg_cost = 0
    #         total_batch = int(train_data.shape[0]/batch_size)
    #         # 循环所有数据
    #         for i in range(total_batch):
    #             batch_xs, batch_ys = iterator_train.get_next()
    #             # 运行优化器
    #             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
    #                                                           y: batch_ys})
    #             # 计算平均loss值
    #             avg_cost += c/total_batch
    #         # 显示训练中的详细信息
    #         if (epoch+1) % display_step == 0:
    #             print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.9f}".format(avg_cost))



