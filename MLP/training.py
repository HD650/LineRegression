from api import *

if __name__ == '__main__':
    training_set, test_set = load_data()
    print('continue training or initialize a new network? y/n')
    c_t = input()
    if c_t is 'y':
        print('reading intermediate result from disk...')
        # 如果之前有训练过的中间结果，读取之后继续训练
        if os.path.exists('./data/weights.dat') and os.path.exists('./data/biases.dat'):
            print('previous learning result found...')
            net = network_reader()
        else:
            print('no previous learning result found...')
            print('using default one...')
            net = Network([784, 30, 10])
        try:
            print('epochs:')
            epochs = int(input())
            print('learning rate:')
            learning_rate = float(input())
            print('batch size:')
            batch_size = int(input())
            print('need test in every epoch? y/n')
            n_t = input()
            print('training...')
            if n_t is 'y':
                net.mini_batch_stochastic_gradient_descent(training_set, epochs, batch_size, learning_rate, test_set)
            else:
                net.mini_batch_stochastic_gradient_descent(training_set, epochs, batch_size, learning_rate, None)
            network_serializer(net)
        except KeyboardInterrupt as e:
            print('end leaning...')
            net.evaluate(test_set)
            network_serializer(net)
    else:
        print('initializing new network...')
        print('please enter the layers of network:')
        layers = int(input())
        structure = list()
        for i in range(layers):
            print('enter number of nodes in layer '+str(i+1))
            structure.append(int(input()))
        net = Network(structure)
        try:
            print('epochs:')
            epochs = int(input())
            print('learning rate:')
            learning_rate = float(input())
            print('batch size:')
            batch_size = int(input())
            print('need test in every epoch? y/n')
            n_t = input()
            print('training...')
            if n_t is 'y':
                net.mini_batch_stochastic_gradient_descent(training_set, epochs, batch_size, learning_rate, test_set)
            else:
                net.mini_batch_stochastic_gradient_descent(training_set, epochs, batch_size, learning_rate, None)
            network_serializer(net)
        except KeyboardInterrupt as e:
            print('end leaning...')
            net.evaluate(test_set)
            network_serializer(net)
