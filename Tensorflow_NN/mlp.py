import tensorflow as tf
import pickle
import gzip
import numpy

# 使用tensorflow和神经网络实现一个softmax分类器


# 读取mnist的数据
def load_data(dataset):
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    return train_set, valid_set, test_set


# 输入层784节点，隐层7840节点，输出层10节点
def init_layer(hidden_layer_number=7840):
    # 输入层是两维向量，第一维不定长代表一个样本，第二维长度784，是输入图片的大小
    sample_x = tf.placeholder(tf.float32, [None, 784], name='sample_x')
    # 样本标签是两维向量，第一维不定长代表一个样本，第二维长度10，是输出的分类
    sample_y = tf.placeholder(tf.float32, [None, 10], name='sample_y')

    # 隐层一层
    # 定义权重矩阵，维度784x10，初始化使用高斯分布的随机矩阵，注意mean是0，如果是1会导致学习很慢
    weight1 = tf.Variable(tf.random_normal([784, hidden_layer_number], name='weights1'))
    # 定义偏移矩阵，是一个10维向量，是高斯矩阵
    bias1 = tf.Variable(tf.random_normal([hidden_layer_number], name='biases1'))
    # 隐层第一层输出
    hidden_layer1 = tf.sigmoid(tf.matmul(sample_x, weight1) + bias1)

    # 输出层
    # 定义权重矩阵，维度784x10，高斯分布矩阵
    weight2 = tf.Variable(tf.random_normal([hidden_layer_number, 10], name='weights2'))
    # 定义偏移矩阵，是一个10维向量，高高斯布矩阵
    bias2 = tf.Variable(tf.random_normal([10], name='biases2'))
    # 隐层第一层输出，注意，因为softmax交叉熵代价函数要求输入是没有经过0-1处理的，所以这一层的输出没有进入sigmoid函数
    output_layer = tf.matmul(hidden_layer1, weight2) + bias2

    # 权重惩罚，防止过拟合的措施
    weight_decay = 0.20 * (tf.reduce_mean(tf.abs(weight1)) + tf.reduce_mean(tf.abs(weight2))
                           + tf.reduce_mean(tf.abs(bias1)) + tf.reduce_mean(tf.abs(bias2)))
    # 代价函数是softmax交叉熵加上权重惩罚
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, sample_y, -1) + weight_decay)
    # 训练方法维梯度下降法
    trainning = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)

    # 返回必要的变量
    return trainning, output_layer, sample_x, sample_y


if __name__ == '__main__':
    # 一次训练使用100个样本
    batch = 100
    # 初始化session
    sess = tf.InteractiveSession()
    trainning, output_layer, sample_x, sample_y = init_layer()
    sess.run(tf.global_variables_initializer())
    # 读取数据
    train_set, valid_set, test_set = load_data('../neural_network/data/mnist.pkl.gz')

    # mnist的label是一个数字，我们把它变成向量
    train_x = train_set[0]
    temp_y = train_set[1]
    train_y = numpy.zeros([temp_y.shape[0], 10])
    for i in range(len(temp_y)):
        index = temp_y[i]
        train_y[i][index] = 1

    valid_x = valid_set[0]
    temp_y = valid_set[1]
    valid_y = numpy.zeros([temp_y.shape[0], 10])
    for i in range(len(temp_y)):
        index = temp_y[i]
        valid_y[i][index] = 1

    # 迭代10000次
    for i in range(10000):
        # 初始化输入，tensorflow里边相当于先定义好函数，之后带入一些值求variable，这里就是定义带入的值
        feed_dict_train = {sample_x: train_x[i * batch:(i + 1) * batch], sample_y: train_y[i * batch:(i + 1) * batch]}
        feed_dict_vaild = {sample_x: valid_x[0:100], sample_y: valid_y[0:100]}
        # run之后所有变量才真的被计算，每次run都需要把函数必要的输入都用feed_dict输进去
        trainning.run(feed_dict=feed_dict_train)
        # 定义一下误差，就是把真实label和output相比，对了就是true不然就是false，之后求平均正确率
        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(sample_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 我们run accuracy这个函数的时候注意了，这个函数的输入还是x和y，但是带入的是valid样本而不是train样本，求出来的是泛化误差
        print(sess.run(accuracy, feed_dict=feed_dict_vaild))

