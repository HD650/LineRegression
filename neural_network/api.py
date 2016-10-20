import numpy
import json
import os
import PIL.Image
from network import Network
import sys
sys.path.append("..")


def load_data():
    """从mnist中读取图像数据，返回的分别是50000条训练数据，10000条测试和核对数据"""
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data()
    # mnist原始数据中，样本的label是int型数字，我们要把它转换成一个10维向量
    training_set = list()
    for x, y in zip(training_data[0], training_data[1]):
        training_set.append((x, mnist_loader.vectorized_result(y)))
    # 同理，转化为向量
    test_set = list()
    for x, y in zip(validation_data[0], validation_data[1]):
        test_set.append((x, mnist_loader.vectorized_result(y)))
    return training_set, test_set


def network_serializer(network):
    """将训练的网络的weights和biases串行化存入磁盘"""
    weights = list()
    print('{0} layers in weights...'.format(len(network.weights)))
    print('serializeing weights matrix...')
    # 将network的每一层的weight矩阵转化为2维list，总体weights是3维list，之后使用json串行化（numpy包内的类型不支持直接串行化）
    for layer in network.weights:
        weight_shape = layer.shape
        print('shape: '+str(weight_shape))
        weights.append(layer.tolist())
    weights = json.dumps(weights)

    biases = list()
    print('{0} layers in biases...'.format(len(network.biases)))
    print('serializeing biases matrix...')
    # 同理，对biases每一层也转化为2维list
    for layer in network.biases:
        bias_shape = layer.shape
        print('shape: '+str(bias_shape))
        biases.append(layer.tolist())
    biases = json.dumps(biases)

    # 写入磁盘
    weight_file = open('./data/weights.dat', 'w')
    bias_file = open('./data/biases.dat', 'w')
    weight_file.write(weights)
    bias_file.write(biases)
    weight_file.close()
    bias_file.close()


def network_reader(network):
    """从磁盘中读取串行化的weights和biases数据，注意调用这个函数会完全覆盖network之前的内部字段，包括他的形状"""
    weight_file = open('./data/weights.dat', 'r')
    bias_file = open('./data/biases.dat', 'r')
    weigths_data = json.loads(weight_file.read())
    biases_data = json.loads(bias_file.read())

    # 文件中的weights是3维list，第一维是层，第二维是矩阵
    weights = list()
    print('found {0} layers in weights...'.format(len(weigths_data)))
    for layer in weigths_data:
        layer_m = numpy.matrix(layer)
        weights.append(layer_m)

    # biases同理是3维list
    biases = list()
    print('found {0} layers in biases...'.format(len(biases_data)))
    for layer in biases_data:
        layer_m = numpy.matrix(layer)
        biases.append(layer_m)

    network.layer_num = len(weights) + 1
    network.weights = weights
    network.biases = biases

    return network


def test_from_image(network, file_name, desired_result):
    """读取一个图片文件，识别其内部文字"""
    image = PIL.Image.open(file_name)
    image.show()
    data_bytes = list(image.getdata())
    input_layer_activation = numpy.matrix(data_bytes, float)
    input_layer_activation /= 255
    output_layer_a = network.forward_feed(input_layer_activation)
    result = output_layer_a.argmax()
    print('network output: '+str(result))
    if result == desired_result:
        print('identify successes!')
        return True
    else:
        print('identify failed!')
        return False


def train_network():
    """debug用"""
    net = Network([784, 30, 10])
    # 读取mnist的训练和测试数据
    training_set, test_set = load_data()
    # 如果之前有训练过的中间结果，读取之后继续训练
    if os.path.exists('./data/weights.dat') and os.path.exists('./data/biases.dat'):
        print('previous learning result found...')
        net = network_reader(net)
    else:
        print('no previous learning result found...')
    try:
        net.mini_batch_stochastic_gradient_descent(training_set, 999999, 100, 3.0, test_set)
        network_serializer(net)
        return net
    except KeyboardInterrupt as e:
        print('end leaning...')
        print('accuracy now: '+str(net.accuracy))
        network_serializer(net)
        return net


def start_job():
    """未完成"""
    net = Network([784, 30, 10])
    # 读取mnist的训练和测试数据
    training_set, test_set = load_data()
    # 如果之前有训练过的中间结果，读取之后继续训练
    if os.path.exists('./data/weights.dat') and os.path.exists('./data/biases.dat'):
        print('previous learning result found...')
        net = network_reader(net)
    else:
        print('no previous learning result found...')
    try:
        net.mini_batch_stochastic_gradient_descent(training_set, 10, 100, 3.0, test_set)
    except KeyboardInterrupt as e:
        print('end leaning...')
        print('accuracy now: '+str(net.accuracy))
        network_serializer(net)