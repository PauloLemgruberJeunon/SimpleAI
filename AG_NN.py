import numpy as np
from sklearn.metrics import mean_squared_error as mqe

TamPop = 10
TaxMut = 1
num_layers = 2
layers_sizes = [2, 1]
feature_qty = 1
maxx = 1
g = 1


# Avaliation function
def nothing(x):
    return np.array(x)


# Avaliation function
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
        neural_net[i] = np.random.uniform(-1, 1, (layers_sizes[i - 1] + 1, layers_sizes[i]))
    return neural_net


# Generate initial population
def initPop():
    ind = np.zeros(TamPop, dtype=np.ndarray)
    for i in range(TamPop):
        ind[i] = create_neural_net(num_layers, layers_sizes, feature_qty)
    return ind


def predict(neural_net, features, activation_func):
    num_layers = neural_net.shape[0]
    result = np.array(np.append(features, 1))
    for i in range(num_layers):
        result = np.matmul(result, neural_net[i])
        result = activation_func(result)
        result = np.append(result, 1)
    result = np.delete(result, -1)
    return result


# Avaliation function
def eval(ind, correct_data):
    fit = np.zeros(TamPop)
    # Temp array used to recive the predicted values from the neural network.
    predictTemp = np.zeros(len(correct_data))
    for i in range(TamPop):  # For each neural network.
        for j in range(len(correct_data)):  # For each value to be predicted.
            predictTemp[j] = predict(ind[i], correct_data[j], nothing)
        # Calculates fit using the mean square error function of the predicted values and the correct values.
        fit[i] = 1 + 1 / mqe(predictTemp, correct_data)

    print('Correct Data: ')
    print(correct_data)
    print('\n')
    print('Data from Neural Network:')
    print(predictTemp.reshape([19, 1]))
    return fit


def crossover(ind, fit):
    maxfit = fit[1]
    maxi = 1
    for i in range(0, TamPop):  # Search for the best. We don't kill the best!
        if (fit[i] > maxfit):
            maxfit = fit[i]
            maxi = i

    for i in range(0, TamPop):
        if(i == maxi):
            continue  # Protect the best.
        # Crossover
        ind[i] = (ind[i] + ind[maxi]) / 2.0  # Using simple mean.
        # Mutation
        ind[i] = ind[i] + ((np.random.uniform(-1, 1) % maxx - (maxx / 2.0)) /
                           100.0) * TaxMut  # Very important!


def square_func(x):
    return [a * a + 5 for a in x]


# Array with correct data collected from a square function.
correct_data = square_func(range(-9, 10))
ind = initPop()  # Start population with random values in a range.

while(1):
    print('\n')
    print('AG Generation: ')
    print(g)

    fit = eval(ind, correct_data)
    crossover(ind, fit)
    g = g + 1

# Ultima aula, n prestei tanta atencao =/
# Quanto mais evoluido estamos, mais dificil eh de evoluir mais.
# Vamos mutar todo mundo, MENOS o melhor de todos. Podemos pensar em diferentes taxas de mutacao caso esteja com erro baixo.
