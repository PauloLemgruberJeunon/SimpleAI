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

    def fit(self, x_array, y_array, epochs):
        predictions = [np.array(self.fit_predict(x), dtype=float) for x in x_array]
        deltas = np.zeros(len(self.layers), dtype=np.ndarray)
        for p in range(len(predictions)):
            deltas[-1] = np.multiply(np.subtract(predictions[p][-1], y_array[p]),
                                     self.layers[-1].activation_fcn.grad(predictions[p][-1]))
            print(deltas)
            quit()
            for l in range(len(self.layers) - 2, -1, -1):
                deltas[l] = np.multiply(np.matmul(self.layers[l].weight_mtr.transpose(), deltas[l + 1]),
                                        self.layers[l].activation_fcn.grad(predictions[p][l]))


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
        return self.activation_fcn.actv((np.matmul(np.append(1, input_array), self.weight_mtr)))


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
