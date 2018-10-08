import numpy as np


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_array):
        result = input_array.copy()
        for i in range(self.layers.shape[0]):
            result = self.layers[i].feed_foward(result)
        return result

    def fit_predict(self, input_array):
        actv_values = np.zeros(len(self.layers) + 1, dtype=np.ndarray)
        actv_values[0] = np.array(input_array, dtype=float)
        for i in range(self.layers.shape[0]):
            actv_values[i + 1] = self.layers[i].feed_foward(actv_values[i])
        return actv_values[1:]

    def fit(self, x_array, y_array, epochs, n):
        deltas = np.zeros(len(self.layers), dtype=np.ndarray)
        for e in range(epochs):
            predictions = [np.array(self.fit_predict(x), dtype=np.ndarray) for x in x_array]
            rights = 0
            for p in range(len(predictions)):
                print('y = ' + str(y_array[p][0]))
                print('predict = ' + str(predictions[p][-1]))
                rights += 1 if np.abs(y_array[p] - predictions[p][-1]) < 0.5 else 0
                deltas[-1] = np.multiply(np.subtract(y_array[p], predictions[p][-1]),
                                         self.layers[-1].activation_fcn.grad(predictions[p][-1]))

                for l in range(len(self.layers) - 2, -1, -1):
                    deltas[l] = np.multiply(np.matmul(self.layers[l + 1].weight_mtr[1:, :], deltas[l + 1]),
                                            self.layers[l].activation_fcn.grad(predictions[p][l]))

                input_with_bias = np.append(1, x_array[p])
                for i in range(self.layers[0].weight_mtr.shape[0]):
                    for j in range(self.layers[0].weight_mtr.shape[1]):
                        self.layers[0].weight_mtr[i][j] = self.layers[0].weight_mtr[i][j] + \
                            (n * deltas[0][j] * input_with_bias[i])

                for l in range(1, len(self.layers)):
                    mtr = self.layers[l].weight_mtr
                    input_with_bias = np.append(1, predictions[p][l - 1])
                    for i in range(mtr.shape[0]):
                        for j in range(mtr.shape[1]):
                            mtr[i][j] = mtr[i][j] + (n * deltas[l][j] * input_with_bias[i])
            print('acc = ' + str(rights / len(predictions)))
            print('\n')


class NeuralLayer:
    def __init__(self, number_of_neurons, actv_fcn, input_size):
        self.activation_fcn = actv_fcn
        self.number_of_neurons = number_of_neurons
        self.input_size = input_size
        # print(self.input_size)
        # print(self.number_of_neurons)
        # print()
        self.weight_mtr = np.random.uniform(-1, 1, (self.input_size + 1, self.number_of_neurons))

    def feed_foward(self, input_array):
        return self.activation_fcn.actv(np.matmul(np.append(1, input_array), self.weight_mtr))


class NeuralFactory:
    def __init__(self):
        pass

    def create_neural_net(self, hidden_actv_fcn, final_actv_fcn, layers_sizes, input_size):
        num_layers = len(layers_sizes)
        layers = np.zeros(num_layers, dtype=np.ndarray)
        layers[0] = NeuralLayer(layers_sizes[0], hidden_actv_fcn, input_size)
        for i in range(1, num_layers - 1):
            layers[i] = NeuralLayer(layers_sizes[i], hidden_actv_fcn, layers_sizes[i - 1])
        layers[num_layers - 1] = NeuralLayer(layers_sizes[-1],
                                             final_actv_fcn, layers_sizes[-2])

        return NeuralNet(layers)


class ActivationFcn:
    def __init__(self):
        pass

    def actv(self, val_array):
        return np.divide(1, np.add(1, np.exp(-val_array)))

    def grad(self, val_array):
        return np.multiply(val_array, np.subtract(1, val_array))
