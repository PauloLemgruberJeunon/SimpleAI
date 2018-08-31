import numpy as np


def sigmoid(x):
    x = np.array(x)
    result = np.empty(x.shape)
    for i in range(x.shape[0]):
        result[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return result


def create_neural_net(num_layers, layers_sizes, feature_qty):
    neural_net = np.zeros(num_layers, dtype=np.ndarray)
    neural_net[0] = np.random.uniform(-1, 1,
                                      (feature_qty + 1, layers_sizes[0]))
    for i in range(1, num_layers):
        neural_net[i] = np.random.uniform(-1, 1,
                                          (layers_sizes[i - 1] + 1, layers_sizes[i]))

    return neural_net


def predict(neural_net, features, activation_func):
    num_layers = neural_net.shape[0]
    result = np.array(np.append(features, 1))
    for i in range(num_layers):
        result = np.matmul(result, neural_net[i])
        result = activation_func(result)
        result = np.append(result, 1)

    result = np.delete(result, -1)
    return result


neural_net = create_neural_net(2, [2, 1], 1)
print(predict(neural_net, 10, sigmoid))
