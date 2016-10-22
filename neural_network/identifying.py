from api import *

if __name__ == '__main__':
    net = network_reader()
    print('start deduction...')
    print('please enter the hand writing directory(absolute directory:)')
    directory = input()
    directory = './data/your_test_set/{0}.bmp'.format(directory)
    print('please enter the desired output:')
    desired_output = int(input())
    test_from_image(net, directory, desired_output)