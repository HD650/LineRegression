import sys
sys.path.append("./")
import tensorflow as tf
import os
from PIL import Image
import numpy as np

SAVER_FILE = "./variables/training_variables"
TRAINING_SET_FILE = "./bin_data/data_batch_%d.bin"
TESTING_SET_FILE = "./bin_data/test_batch.bin"
MAX_STEPS = 10000 * 4
LEARNING_RATE = 1e-4
DROPOUT_PROB = 0.5
BATCH_SIZE = 100
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 10000


def read_test_image(batch_size):
    file_in = TESTING_SET_FILE
    files = tf.train.string_input_producer([file_in])
    reader = tf.FixedLengthRecordReader(record_bytes=1+32*32*3)
    key, value = reader.read(files)
    data = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.slice(data, [0], [1]), tf.int64)
    raw_image = tf.reshape(tf.slice(data, [1], [32*32*3]), [3, 32, 32])
    image = tf.cast(tf.transpose(raw_image, [1, 2, 0]), tf.float32)

    std_image = tf.image.per_image_standardization(image)

    # 将上边的单一tensor打包，返回的是4D tensor了，每当training中读取（依赖）这个4D tensor，这里的operator就会跑，使得
    # 样本被填充，同时还会有其他线程继续深入到上边那些代码，从二进制文件中读出新的样本并预处理，然后填入这个buffer
    # 这个打包函数会创建一个queue buffer，这个buffer最大是4000+3batch size，一定会维持起码4000个样本在里边,取4000会使得
    # 其shuffle效果比较好
    # 注意，在定义好读取数据的graph之后，真正run这些tensor之前，应该tf.train.start_queue_runners(sess=sess)
    images, labels = tf.train.shuffle_batch([std_image, label],
                           batch_size=batch_size,
                           num_threads=16,
                           capacity=int(TEST_SET_SIZE * 0.4 + 3 * batch_size),
                           min_after_dequeue=int(TEST_SET_SIZE * 0.4)
                                    )
    # 返回的两个tensor一个是[batch_size, wight, height, channel]，一个是[batch_size]，注意第二个是1D tensor
    return images, tf.reshape(labels, [-1])


def read_distorted_image(batch_size):
    files_in = [TRAINING_SET_FILE % i for i in range(1, 6)]
    files = tf.train.string_input_producer(files_in)
    reader = tf.FixedLengthRecordReader(record_bytes=1+32*32*3)
    key, value = reader.read(files)
    data = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.slice(data, [0], [1]), tf.int64)
    raw_image = tf.reshape(tf.slice(data, [1], [32*32*3]), [3, 32, 32])
    image = tf.cast(tf.transpose(raw_image, [1, 2, 0]), tf.float32)

    lr_image = tf.image.random_flip_left_right(image)
    # 改变亮度的delta 是255/4
    br_image = tf.image.random_brightness(lr_image, max_delta=63)
    rc_image = tf.image.random_contrast(br_image, lower=0.2, upper=1.8)

    std_image = tf.image.per_image_standardization(rc_image)

    # 将上边的单一tensor打包，返回的是4D tensor了，每当training中读取（依赖）这个4D tensor，这里的operator就会跑，使得
    # 样本被填充，同时还会有其他线程继续深入到上边那些代码，从二进制文件中读出新的样本并预处理，然后填入这个buffer
    # 这个打包函数会创建一个queue buffer，这个buffer最大是4000+3batch size，一定会维持起码4000个样本在里边,取4000会使得
    # 其shuffle效果比较好
    # 注意，在定义好读取数据的graph之后，真正run这些tensor之前，应该tf.train.start_queue_runners(sess=sess)
    images, labels = tf.train.shuffle_batch([std_image, label],
                           batch_size=batch_size,
                           num_threads=16,
                           capacity=int(TRAIN_SET_SIZE * 0.4 + 3 * batch_size),
                           min_after_dequeue=int(TRAIN_SET_SIZE * 0.4)
                                    )
    return images, tf.reshape(labels, [-1])


class CNN(object):

    def __init__(self):
        self.sess = tf.InteractiveSession()

    def initialize_network(self):
        # 输入层
        self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.int64, [None])
        # 第一卷基层，使用5*5卷积核，输出32张feature map
        self.conv_w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.conv_b1 = tf.Variable(tf.constant(0.1, tf.float32, [32]))
        self.conv_l1 = tf.nn.relu(tf.nn.conv2d(self.images, self.conv_w1, [1, 1, 1, 1], padding="SAME") + self.conv_b1)
        # 参照AlexNet的overlap pooling，能稍微防止过拟合，这里暂时使用传统pooling
        self.pool_l1 = tf.nn.max_pool(self.conv_l1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        # 参照AlexNet的local response normalization结构，不过第一次norm在pooling之后，第二次在之前
        self.lrn_l1 = tf.nn.lrn(self.pool_l1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        # 第二卷基层，连接32张feature map，使用5*5卷积核输出64张feature map
        self.conv_w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.conv_b2 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
        self.conv_l2 = tf.nn.relu(tf.nn.conv2d(self.lrn_l1, self.conv_w2, [1, 1, 1, 1], padding="SAME") + self.conv_b2)
        self.lrn_l2 = tf.nn.lrn(self.conv_l2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.pool_l2 = tf.nn.max_pool(self.lrn_l2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

        # 第三卷基层，连接64张feature map，使用5*5卷积核输出64张feature map
        self.conv_w3 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.conv_b3 = tf.Variable(tf.constant(0.1, tf.float32, [64]))
        self.conv_l3 = tf.nn.relu(tf.nn.conv2d(self.pool_l2, self.conv_w3, [1, 1, 1, 1], padding="SAME") + self.conv_b3)
        self.pool_l3 = tf.nn.max_pool(self.conv_l3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
        # 将64张4*4的feature map拉伸为一维向量
        self.pool_l3_flat = tf.reshape(self.pool_l3, [-1, 4*4*64])

        # 第一全连接层，384个隐节点
        self.full_w1 = tf.Variable(tf.truncated_normal([4*4*64, 384], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.full_b1 = tf.Variable(tf.constant(0.1, tf.float32, [384]))
        self.full_l1 = tf.nn.relu(tf.matmul(self.pool_l3_flat, self.full_w1) + self.full_b1)

        # 在全连阶层加上dropout防止过拟合
        self.drop_prob = tf.placeholder(tf.float32)
        # 注意drop_prob是保留输出的概率，不是被舍弃的概率，同时，因为有概率被舍弃，则输出也要相应变大
        self.full_l1_drop = tf.nn.dropout(self.full_l1, self.drop_prob)

        # 第二全链接层
        self.full_w2 = tf.Variable(tf.truncated_normal([384, 192], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.full_b2 = tf.Variable(tf.constant(0.1, tf.float32, [192]))
        self.full_l2 = tf.nn.relu(tf.matmul(self.full_l1_drop, self.full_w2) + self.full_b2)
        self.full_l2_drop = tf.nn.dropout(self.full_l2, self.drop_prob)

        # 全链接输出层，使用softfmax进行分类
        self.full_w3 = tf.Variable(tf.truncated_normal([192, 10], stddev=5e-4, mean=0.0), dtype=tf.float32)
        self.full_b3 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
        self.full_l3 = tf.matmul(self.full_l2_drop, self.full_w3) + self.full_b3

        # 代价函数使用cross entropy
        self.cost_function = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(self.full_l3, self.labels)))
        self.training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost_function)

        # 计算正确率
        self.right_answer = tf.equal(tf.argmax(self.full_l3, axis=1), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.right_answer, tf.float32))

        self.summary_accuracry = tf.summary.scalar("Accuracy", self.accuracy)
        self.summary_loss_train = tf.summary.scalar("Training Loss", self.cost_function)
        self.summary_loss_test = tf.summary.scalar("Testing Loss", self.cost_function)
        self.summary_cov_w1 = tf.summary.histogram("L1 Conv Weights", self.conv_w1)
        self.summary_cov_w2 = tf.summary.histogram("L2 Conv Weights", self.conv_w2)
        self.summary_cov_w3 = tf.summary.histogram("L3 Conv Weights", self.conv_w3)
        self.summary_cov_b1 = tf.summary.histogram("L1 Conv Biases", self.conv_b1)
        self.summary_cov_b2 = tf.summary.histogram("L2 Conv Biases", self.conv_b2)
        self.summary_cov_b3 = tf.summary.histogram("L3 Conv Biases", self.conv_b3)
        self.summary_cov_kernel1 = tf.summary.image("L1 Conv kernel",
                                                    tf.transpose(self.conv_w1, [3, 0, 1, 2]), max_outputs=32)
        self.summary_images = tf.summary.image("Input Image",
                                               tf.slice(self.images, [0, 0, 0, 0], [1, 32, 32, 3]), max_outputs=1)
        self.summary_conv_image1 = tf.summary.image("L1 Conv Output", tf.transpose(
            tf.slice(self.conv_l1, [0, 0, 0, 0], [1, 32, 32, 32]), [3, 1, 2, 0]), max_outputs=32)

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter("./record", self.sess.graph)

    def train(self):
        try:
            self.initialize_network()
            if os.path.exists(SAVER_FILE + ".index"):
                print("saved variables found, load it? y/n")
                is_load = input()
                if is_load is "y":
                    self.saver.restore(self.sess, SAVER_FILE)
                else:
                    self.sess.run(tf.global_variables_initializer())
            else:
                self.sess.run(tf.global_variables_initializer())

            training_images, training_labels = read_distorted_image(BATCH_SIZE)
            testing_images, testing_labels = read_test_image(BATCH_SIZE)
            tf.train.start_queue_runners(sess=self.sess)

            for step in range(MAX_STEPS):
                print("training"+str(step))
                i, l = self.sess.run([training_images, training_labels])
                _, loss = self.sess.run([self.training, self.summary_loss_train], feed_dict={self.images: i,
                                                        self.labels: l,
                                                        self.drop_prob: DROPOUT_PROB})
                self.writer.add_summary(loss, step)
                if step % 10 is 0:
                    print("validation")
                    i, l = self.sess.run([testing_images, testing_labels])
                    ac, sum_ac, sum_loss = self.sess.run([self.accuracy, self.summary_accuracry, self.summary_loss_test],
                                       feed_dict={self.images: i,
                                                self.labels: l,
                                                self.drop_prob: 1.0})
                    self.writer.add_summary(sum_ac, step)
                    self.writer.add_summary(sum_loss, step)
                    print(str(ac))

                if step % 100 is 0:
                    i, l = self.sess.run([testing_images, testing_labels])
                    w1, b1, w2, b2 = self.sess.run([self.summary_cov_w1, self.summary_cov_b1,
                                                    self.summary_cov_w2, self.summary_cov_b2],
                                       feed_dict={self.images: i,
                                                self.labels: l,
                                                self.drop_prob: 1.0})
                    self.writer.add_summary(w1, step)
                    self.writer.add_summary(b1, step)
                    self.writer.add_summary(w2, step)
                    self.writer.add_summary(b2, step)

                if step % 1000 is 0:
                    i, l = self.sess.run([testing_images, testing_labels])
                    image, activate, kernel = self.sess.run([self.summary_images, self.summary_conv_image1, self.summary_cov_kernel1],
                                       feed_dict={self.images: i,
                                                self.labels: l,
                                                self.drop_prob: 1.0})
                    self.writer.add_summary(image, step)
                    self.writer.add_summary(activate, step)
                    self.writer.add_summary(kernel, step)
            self.saver.save(self.sess, SAVER_FILE)
        except KeyboardInterrupt as e:
            print(str(e))
            self.saver.save(self.sess, SAVER_FILE)
            print("saved")
            input()

if __name__ == '__main__':
    network = CNN()
    network.train()