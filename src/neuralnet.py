import numpy as np


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.delta_matrices = self.init_delta_matrices()

    def init_delta_matrices(self):
        delta_matrices = np.array([np.zeros(self.layers[i].weight_mtr.shape, dtype=float)
                                   for i in range(self.layers.shape[0])], dtype=np.ndarray)
        return delta_matrices

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
            deltas[-1] = np.subtract(predictions[p][-1], y_array[p])
            for l in range(len(self.layers) - 2, -1, -1):
                deltas[l] = np.multiply(np.matmul(self.layers[l].weight_mtr.transpose(), deltas[l + 1]),
                                        self.layers[l].activation_fcn.grad(predictions[p][l]))


class NeuralLayer:
    def __init__(self, number_of_neurons, actv_fcn, input_size):
        self.activation_fcn = actv_fcn
        self.number_of_neurons = number_of_neurons
        self.input_size = input_size
        self.weight_mtr = np.random.uniform(-1, 1, (self.input_size + 1, self.number_of_neurons))

    def feed_foward(self, input_array):
        return self.activation_fcn(np.matmul(np.append(input_array, 1), self.weight_mtr))


class NeuralFactory:
    def __init__(self):
        pass

    def create_neural_net(self, hidden_actv_fcn, final_actv_fcn, layers_sizes, input_size):
        num_layers = len(layers_sizes)
        layers = np.zeros(num_layers, dtype=np.ndarray)
        layers[0] = NeuralLayer(layers_sizes[0], hidden_actv_fcn, input_size)
        for i in range(1, num_layers - 1):
            layers[i] = NeuralLayer(layers_sizes[i], hidden_actv_fcn, layers_sizes[i - 1])
        layers[num_layers - 1] = NeuralLayer(layers_sizes[num_layers - 1],
                                             final_actv_fcn, layers_sizes[num_layers - 1])
        return NeuralNet(layers)


class ActivationFcn:
    pass
