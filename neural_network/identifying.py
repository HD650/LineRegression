from api import *

if __name__ == '__main__':
    print('need training at first:')
    is_training = input()
    if is_training is 'true' or is_training is 'True' or is_training is '1':
        net = train_network()
        print('training end...')
    else:
        net = Network([784, 30, 10])
        net = network_reader(net)
    print('start deduction...')
    print('please enter the hand writing directory(absolute directory:)')
    directory = input()
    directory = './data/your_test_set/{0}.bmp'.format(directory)
    print('please enter the desired output:')
    desired_output = int(input())
    test_from_image(net, directory, desired_output)