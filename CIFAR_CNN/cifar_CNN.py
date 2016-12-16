import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import image_pro
import matplotlib.pyplot as plt
import os

saver_file_name = "./variables/training_variables"
training_file = "./data/reorganized_batch"
test_file = "./data/reorganized_test_batch"
training_epchos = 10000
learning_rate = 1e-5
dropout_rate = 0.9
batch_size = 100


def draw_convolutional_kernel(data):
    plt.ion()
    fig = plt.figure()
    plot_count = data.shape[-1]
    axs = list()
    for i in range(plot_count):
        temp_ax = fig.add_subplot(4, 8, i+1)
        temp_ax.imshow(data[:, :, :, i])
        axs.append(temp_ax)
    plt.draw()
    pass


# 三层5*5卷积层，每层2*2池化，最后接一层4096全链接层，最终输出10分类
# 调试参数：学习率不能高于1e－5，1e－4的时候学习率太高loss会来回震荡
def initialize_network():
    # 输入层为3072维向量
    raw_input = tf.placeholder(tf.float32, [None, 3072])
    sample_y = tf.placeholder(tf.float32, [None, 10])
    sample_x = tf.reshape(raw_input, [-1, 32, 32, 3])

    # 第一卷基层，使用5*5卷积核，输出32张feature map
    conv_w1 = tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.1), dtype=tf.float32)
    conv_b1 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    conv_l1 = tf.nn.relu(tf.nn.conv2d(sample_x, conv_w1, [1, 1, 1, 1], padding="SAME") + conv_b1)
    pool_l1 = tf.nn.max_pool(conv_l1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    # 第二卷基层，连接32张feature map，使用5*5卷积核输出64张feature map
    conv_w2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1), dtype=tf.float32)
    conv_b2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l2 = tf.nn.relu(tf.nn.conv2d(pool_l1, conv_w2, [1, 1, 1, 1], padding="SAME") + conv_b2)
    pool_l2 = tf.nn.max_pool(conv_l2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    # 第三卷基层，连接64张feature map，使用5*5卷积核输出64张feature map
    conv_w3 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=0.1), dtype=tf.float32)
    conv_b3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l3 = tf.nn.relu(tf.nn.conv2d(pool_l2, conv_w3, [1, 1, 1, 1], padding="SAME") + conv_b3)
    pool_l3 = tf.nn.max_pool(conv_l3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    # 将64张4*4的feature map拉伸为一维向量
    pool_l3_flat = tf.reshape(pool_l3, [-1, 4*4*64])

    # 全连接层，4096个隐节点
    full_w1 = tf.Variable(tf.random_normal([4*4*64, 4096], stddev=0.1), dtype=tf.float32)
    full_b1 = tf.Variable(tf.constant(0.1, tf.float32, [4096]))
    full_l1 = tf.nn.relu(tf.matmul(pool_l3_flat, full_w1) + full_b1)

    # 在全连阶层加上dropout防止过拟合
    drop_prob = tf.placeholder(tf.float32)
    # 注意drop_prob是保留输出的概率，不是被舍弃的概率，同时，因为有概率被舍弃，则输出也要相应变大
    full_l1_drop = tf.nn.dropout(full_l1, drop_prob)

    # 全链接输出层，使用softfmax进行分类
    full_w2 = tf.Variable(tf.random_normal([4096, 10]), dtype=tf.float32)
    full_b2 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    full_l2 = tf.matmul(full_l1_drop, full_w2) + full_b2

    # 代价函数使用cross entropy
    cost_function = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(full_l2, sample_y)))
    training = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    # 计算正确率
    right_answer = tf.equal(tf.argmax(full_l2, axis=1), tf.argmax(sample_y, axis=1))
    right_rate = tf.reduce_mean(tf.cast(right_answer, tf.float32))

    # 返回必要的tensor
    return raw_input, sample_y, training, right_rate, drop_prob, \
        [conv_l1, conv_l2, conv_l3,  full_l1, full_l2], [conv_w1, conv_w2, conv_w3]


def training(epchos):
    # 初始化tensorflow相关的模块
    print("initializing session...")
    sess = tf.InteractiveSession()
    print("initializing network...")
    x, y, train, correct_rate, dropout, layers, convs = initialize_network()
    saver = tf.train.Saver()

    # 读取之前的网络信息
    if os.path.exists(saver_file_name+".index"):
        print("saved variables found, load it? y/n")
        is_load = input()
        if is_load is "y":
            print("loading saved training variables...")
            saver.restore(sess, saver_file_name)
        else:
            print("initializing new variables...")
            sess.run(tf.global_variables_initializer())
    else:
        print("initializing new variables...")
        sess.run(tf.global_variables_initializer())

    # 初始化数据
    training_set = list()
    for i in range(5):
        temp_training_set = image_pro.open_pickled_data(training_file+str(i+1))
        training_set.append(temp_training_set)
        testing_set = image_pro.open_pickled_data(test_file)
    size = len(temp_training_set["labels"])

    # 开始训练，并且监听键盘中断
    try:
        for epcho in range(epchos):
            # 产生batch的随机数
            index = np.random.random_integers(0, size / batch_size) - 1
            begin = index*batch_size
            end = (index+1)*batch_size
            # 从总样本中提取一小batch
            training_x = training_set[epcho % 5]["data"][begin:end, :]
            training_y = training_set[epcho % 5]["labels"][begin:end, :]
            testing_x = testing_set["data"][begin:end, :]
            testing_y = testing_set["labels"][begin:end, :]
            # 一次训练迭代
            sess.run(train, feed_dict={x: training_x, y: training_y, dropout: dropout_rate})
            # 一次检测迭代
            valid = sess.run(correct_rate, feed_dict={x: testing_x, y: testing_y, dropout: 1.0})
            if valid is "nan":
                pass
            print("epcho {1} accuracy: {0}".format(valid, str(epcho)))
    # 发生键盘中断之后，停止训练并储存Variables
    except KeyboardInterrupt as ki:
        print(str(ki))
        print("training has been interrupt, saving variables...")
        path = saver.save(sess, saver_file_name)
        print("variables saved in {0}.".format(path))
        conv_value = sess.run(convs)
        draw_convolutional_kernel(conv_value[0])
        print("press any key to exit.")
        input()
        exit()


def predict(filename):
    print("initializing session...")
    sess = tf.InteractiveSession()

    print("initializing network...")
    x, y, train, correct_rate, dropout, layers = initialize_network()
    saver = tf.train.Saver()

    if os.path.exists(saver_file_name):
        print("reading network variables...")
        saver.restore(sess, saver_file_name)
    else:
        print("no network has been trained, quiting...")
        return


if __name__ == '__main__':
    training(training_epchos)




