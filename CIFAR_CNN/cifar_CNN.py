import sys
sys.path.append("./")
import tensorflow as tf
import numpy as np
import image_pro
import matplotlib.pyplot as plt
import os

saver_file_name = "./training_variables"
plt.ion()
fig = plt.figure()


def draw_convolutional_kernel(data):
    pass


# 三层5*5卷积层，每层2*2池化，最后接一层4096全链接层，最终输出10分类
# 调试参数：学习率不能高于1e－5，1e－4的时候学习率太高loss会来回震荡
def initalize_network():
    #
    raw_input = tf.placeholder(tf.float32, [None, 3072])
    sample_y = tf.placeholder(tf.float32, [None, 10])
    sample_x = tf.reshape(raw_input, [-1, 32, 32, 3])

    #
    conv_w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.1), dtype=tf.float32)
    conv_b1 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
    conv_l1 = tf.nn.relu(tf.nn.conv2d(sample_x, conv_w1, [1, 1, 1, 1], padding="SAME") + conv_b1)
    pool_l1 = tf.nn.max_pool(conv_l1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    #
    conv_w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), dtype=tf.float32)
    conv_b2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l2 = tf.nn.relu(tf.nn.conv2d(pool_l1, conv_w2, [1, 1, 1, 1], padding="SAME") + conv_b2)
    pool_l2 = tf.nn.max_pool(conv_l2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    #
    conv_w3 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.1), dtype=tf.float32)
    conv_b3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
    conv_l3 = tf.nn.relu(tf.nn.conv2d(pool_l2, conv_w3, [1, 1, 1, 1], padding="SAME") + conv_b3)
    pool_l3 = tf.nn.max_pool(conv_l3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

    pool_l3_flat = tf.reshape(pool_l3, [-1, 4*4*64])

    #
    full_w1 = tf.Variable(tf.truncated_normal([4*4*64, 4096], stddev=0.1), dtype=tf.float32)
    full_b1 = tf.Variable(tf.constant(0.1, tf.float32, [4096]))
    full_l1 = tf.nn.relu(tf.matmul(pool_l3_flat, full_w1) + full_b1)

    # 在全连阶层加上dropout防止过拟合
    drop_prob = tf.placeholder(tf.float32)
    # 注意drop_prob是保留输出的概率，不是被舍弃的概率，同时，因为有概率被舍弃，则输出也要相应变大
    full_l1_drop = tf.nn.dropout(full_l1, drop_prob)

    #
    full_w2 = tf.Variable(tf.truncated_normal([4096, 10], stddev=0.1), dtype=tf.float32)
    full_b2 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    full_l2 = tf.matmul(full_l1_drop, full_w2) + full_b2

    #
    cost_function = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(full_l2, sample_y)))
    training = tf.train.AdamOptimizer(1e-5).minimize(cost_function)

    #
    right_answer = tf.equal(tf.argmax(full_l2, axis=1), tf.argmax(sample_y, axis=1))
    right_rate = tf.reduce_mean(tf.cast(right_answer, tf.float32))

    return raw_input, sample_y, training, right_rate, drop_prob, \
           [conv_l1, conv_l2, conv_l3, full_l1, full_l2], [conv_w1, conv_w2, conv_w3]


def training(epchos):
    print("initializing session...")
    sess = tf.InteractiveSession()

    print("initializing network...")
    x, y, train, correct_rate, dropout, layers, convs = initalize_network()
    saver = tf.train.Saver()

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

    training_set = image_pro.open_pickled_data("./data/reorganized_batch1")
    testing_set = image_pro.open_pickled_data("./data/reorganized_test_batch")
    size = len(training_set["labels"])
    batch = 100

    try:
        for epcho in range(epchos):
            index = np.random.random_integers(0, size/batch) - 1
            training_x = training_set["data"][index*batch:index*batch+batch, :]
            training_y = training_set["labels"][index*batch:index*batch+batch, :]
            testing_x = testing_set["data"][index*batch:index*batch+batch, :]
            testing_y = testing_set["labels"][index*batch:index*batch+batch, :]

            sess.run(train, feed_dict={x: training_x, y: training_y, dropout: 0.9})
            valid = sess.run([convs, correct_rate], feed_dict={x: testing_x, y: testing_y, dropout: 1.0})
            print(str(valid[0]))
            print("accuracy: {0}".format(valid[1]))
    except KeyboardInterrupt as ki:
        print(str(ki))
        print("training has been interrupt, saving variables...")
        path = saver.save(sess, saver_file_name)
        print("variables saved in {0}.".format(path))
        # sess.run(convs)
        # draw_convolutional_kernel(convs[0])
        input()
        exit()


def predict(filename):
    print("initializing session...")
    sess = tf.InteractiveSession()

    print("initializing network...")
    x, y, train, correct_rate, dropout, layers = initalize_network()
    saver = tf.train.Saver()

    if os.path.exists(saver_file_name):
        print("reading network variables...")
        saver.restore(sess, saver_file_name)
    else:
        print("no network has been trained, quiting...")
        return


if __name__ == '__main__':
    training(10000)




