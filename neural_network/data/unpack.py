import pickle
import gzip
import os
from PIL import Image
# 用于把mnist中的压缩图片数据解包


def save_image(raw_data, dir_name, count):
    if not os.path.exists('./{0}'.format(dir_name)):
        print('no directory found, create one...')
        os.mkdir('./{0}'.format(dir_name))

    for i, (x, y) in enumerate(zip(raw_data[0], raw_data[1])):
        x *= 255
        x = x.astype(int)
        # 将ndarray转换为list之后再转为bytes就不会出错，直接用ndarray的tobytes传出来的字节流并不正确
        # 原因未知,大概是因为numpy中矩阵内有冗余数据,或者其内存分布并不是数学上的顺序
        x = list(x)
        x = bytes(x)
        if i > count:
            break
        image = Image.frombytes('L', (28, 28), x)
        image.save('./{2}/{0}_{1}.bmp'.format(str(i), str(y), dir_name))
        print('./{2}/{0}_{1}.bmp'.format(str(i), str(y), dir_name))


if __name__ == '__main__':
    print('reading mnist.pkl.gz ...')
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    print('{0} training samples found.'.format(len(training_data[0])))
    print('please enter the beginning index you want of the training samples:')
    index = int(input())
    print('how many traning samples to unpack:')
    count = int(input())
    training_set = list()
    training_set.append(training_data[0][index:])
    training_set.append(training_data[1][index:])
    save_image(training_set, 'mnist_training_set', count)
    print('{0} test samples found.'.format(len(test_data[0])))
    print('please enter the beginning index you want of the test samples:')
    index = int(input())
    print('how many test samples to unpack:')
    count = int(input())
    test_set = list()
    test_set.append(test_data[0][index:])
    test_set.append(test_data[1][index:])
    save_image(test_data, 'mnist_test_set', count)
    print('all done.')