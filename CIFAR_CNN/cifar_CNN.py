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
training_epchos = 10000 * 50
learning_rate = 1e-4
dropout_rate = 0.5
batch_size = 100


# 绘制出第一层卷积层的激励
def draw_image(data):
    plt.ioff()
    fig = plt.figure()
    plot_count = data.shape[-1]
    for i in range(plot_count):
        temp_ax = fig.add_subplot(4, 8, i+1)
        temp_ax.imshow(data[:, :, i], "gray")
        # temp_ax.
    plt.show()
    pass


# 三层5*5卷积层，每层2*2池化，最后接一层4096全链接层，最终输出10分类
# 调试参数：学习率1e-4可以一直上升至70正确率
def initialize_network():
    # 输入层为3072维向量
    raw_input = tf.placeholder(tf.float32, [None, 3072])
    sample_y = tf.placeholder(tf.float32, [None, 10])
    sample_x = tf.reshape(raw_input, [-1, 32, 32, 3])

    # 第一卷基层，使用5*5卷积核，输出32张feature map
    conv_w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=5e-2, mean=0.1), dtype=tf.float32)
    conv_b1 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    conv_l1 = tf.nn.relu(tf.nn.conv2d(sample_x, conv_w1, [1, 1, 1, 1], padding="SAME") + conv_b1)
    # 参照AlexNet的overlap pooling，能稍微防止过拟合，这里暂时使用传统pooling
    pool_l1 = tf.nn.max_pool(conv_l1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    # 参照AlexNet的local response normalization结构，不过第一次norm在pooling之后，第二次在之前
    lrn_l1 = tf.nn.lrn(pool_l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # 第二卷基层，连接32张feature map，使用5*5卷积核输出64张feature map
    conv_w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2, mean=0.1), dtype=tf.float32)
    conv_b2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l2 = tf.nn.relu(tf.nn.conv2d(lrn_l1, conv_w2, [1, 1, 1, 1], padding="SAME") + conv_b2)
    lrn_l2 = tf.nn.lrn(conv_l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool_l2 = tf.nn.max_pool(lrn_l2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    # 第三卷基层，连接64张feature map，使用5*5卷积核输出64张feature map
    conv_w3 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-2, mean=0.1), dtype=tf.float32)
    conv_b3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l3 = tf.nn.relu(tf.nn.conv2d(pool_l2, conv_w3, [1, 1, 1, 1], padding="SAME") + conv_b3)
    pool_l3 = tf.nn.max_pool(conv_l3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    # 将64张4*4的feature map拉伸为一维向量
    pool_l3_flat = tf.reshape(pool_l3, [-1, 4*4*64])

    # 第一全连接层，384个隐节点
    full_w1 = tf.Variable(tf.truncated_normal([4*4*64, 384], stddev=5e-2, mean=0.1), dtype=tf.float32)
    full_b1 = tf.Variable(tf.constant(0.1, tf.float32, [384]))
    full_l1 = tf.nn.relu(tf.matmul(pool_l3_flat, full_w1) + full_b1)

    # 在全连阶层加上dropout防止过拟合
    drop_prob = tf.placeholder(tf.float32)
    # 注意drop_prob是保留输出的概率，不是被舍弃的概率，同时，因为有概率被舍弃，则输出也要相应变大
    full_l1_drop = tf.nn.dropout(full_l1, drop_prob)

    # 第二全链接层
    full_w2 = tf.Variable(tf.truncated_normal([384, 192], stddev=5e-2, mean=0.1), dtype=tf.float32)
    full_b2 = tf.Variable(tf.constant(0.1, tf.float32, [192]))
    full_l2 = tf.nn.relu(tf.matmul(full_l1_drop, full_w2) + full_b2)
    full_l2_drop = tf.nn.dropout(full_l2, drop_prob)

    # 全链接输出层，使用softfmax进行分类
    full_w3 = tf.Variable(tf.truncated_normal([192, 10], stddev=5e-2, mean=0.1), dtype=tf.float32)
    full_b3 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    full_l3 = tf.matmul(full_l2_drop, full_w3) + full_b3

    # 代价函数使用cross entropy
    cost_function = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(full_l3, sample_y)))
    training = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    # 计算正确率
    right_answer = tf.equal(tf.argmax(full_l3, axis=1), tf.argmax(sample_y, axis=1))
    right_rate = tf.reduce_mean(tf.cast(right_answer, tf.float32))

    # 记录cost function输出和accuracy
    loss = tf.summary.scalar("loss", cost_function)
    ac_train = tf.summary.scalar("accuracy_training", right_rate)
    ac_test = tf.summary.scalar("accuracy_testing", right_rate)
    conv_l1_W = tf.summary.histogram("Conv_L1_W", conv_w1)
    conv_l1_b = tf.summary.histogram("Conv_L1_B", conv_b1)
    output = tf.summary.histogram("Output", tf.nn.softmax(full_l3))
    # 返回必要的tensor
    return raw_input, sample_y, training, right_rate, drop_prob, \
        [conv_l1, conv_l2, conv_l3,  full_l1, full_l2], loss, ac_train, ac_test, conv_l1_W, conv_l1_b, output


def training(epchos):
    # 初始化tensorflow相关的模块
    print("initializing session...")
    sess = tf.InteractiveSession()
    print("initializing network...")
    x, y, train, correct_rate, dropout, layers, loss, ac_train, ac_test, conv_w, conv_b, output = initialize_network()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter("./record", sess.graph)

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
    size = 0
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
            print("{0} samples in batch{1}".format(str(batch_size), str(epcho % 5)))
            print("samples begin with {0} end to {1}".format(str(begin), str(end)))
            training_x = training_set[epcho % 5]["data"][begin:end, :]
            training_y = training_set[epcho % 5]["labels"][begin:end, :]
            testing_x = testing_set["data"][begin:end, :]
            testing_y = testing_set["labels"][begin:end, :]
            if len(training_x) is not batch_size:
                print("Error occur when obtaining samples, continue...")
                continue
            # 一次训练迭代
            loss_s, ac_train_s, _ = sess.run([loss, ac_train, train], feed_dict={x: training_x, y: training_y, dropout: dropout_rate})
            train_writer.add_summary(loss_s, epcho)
            train_writer.add_summary(ac_train_s, epcho)
            # 一次检测迭代
            ac, layers_r, ac_test_s, conv_w_s, conv_b_s = sess.run([correct_rate, layers, ac_test, conv_w, conv_b], feed_dict={x: testing_x, y: testing_y, dropout: 1.0})
            print("epcho {1} accuracy: {0}".format(ac, str(epcho)))
            train_writer.add_summary(ac_test_s, epcho)
            train_writer.add_summary(conv_w_s, epcho)
            train_writer.add_summary(conv_b_s, epcho)
        print("training over...")
        path = saver.save(sess, saver_file_name)
        print("variables saved in {0}.".format(path))
    # 发生键盘中断之后，停止训练并储存Variables
    except KeyboardInterrupt as ki:
        print(str(ki))
        print("training has been interrupt, saving variables...")
        path = saver.save(sess, saver_file_name)
        print("variables saved in {0}.".format(path))
        layers_value = sess.run(layers, feed_dict={x: testing_x, y: testing_y, dropout: 1.0})
        layers_value = layers_value[0][0, :, :, :]
        layers_value = layers_value / layers_value.max() * 254
        draw_image(layers_value)
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




