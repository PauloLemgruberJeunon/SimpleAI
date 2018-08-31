import numpy as np
from sklearn.metrics import mean_squared_error as mqe

TamPop = 10
TaxMut = 0.01
num_layers = 2
layers_sizes = [2, 1]
feature_qty = 1

def nothing(x):
	return np.array(x)

def sigmoid(x):
    x = np.array(x)
    result = np.empty(x.shape)
    for i in range(x.shape[0]):
        result[i] = 1.0 / (1.0 + np.exp(-x[i]))
    return result


def create_neural_net(num_layers, layers_sizes, feature_qty):
	neural_net = np.zeros(num_layers, dtype=np.ndarray)
	neural_net[0] = np.random.uniform(-1, 1, (feature_qty + 1, layers_sizes[0]))

	for i in range(1, num_layers):
		neural_net[i] = np.random.uniform(-1, 1,(layers_sizes[i - 1] + 1, layers_sizes[i]))
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

def initPop():
    ind = np.zeros(TamPop, dtype=np.ndarray)
    for i in range(TamPop):
        ind[i] = create_neural_net(num_layers, layers_sizes, feature_qty)
    return ind

def square_func(x):
	return [a*a+5 for a in x]

def eval(ind):
	fit = np.zeros(TamPop)
	for i in range(TamPop):
		fit[i] = predict(ind[i], 10, nothing)
	return fit

def crossover(fit):
	maxfit = fit[1]
	maxi   = 1
	for i in range(0, TamPop):  # Busca pelo melhor individuo
		if (fit[i] > maxfit):
			maxfit = fit[i]
			maxi = i

correct_data = square_func(range(-9,10))
ind = initPop()
fit = eval(ind)
crossover(fit)

#Quanto mais evoluido estamos, mais dificil eh de evoluir mais.
#Vamos mutar todo mundo, MENOS o melhor de todos.