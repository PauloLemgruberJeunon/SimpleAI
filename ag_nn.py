import numpy as np
import heapq
from sklearn.metrics import mean_squared_error as mqe
np.set_printoptions(precision=2)

TAM_POP = 20
TAX_MUT = 0.05
LAYERS_SIZES = [2, 2, 1]
FEATURE_QTY = 1
MAXX = 1
G = 1


def math_function(x_array):
    return [(x ** 2) + 5 for x in x_array]


def relu(values_array):
    return np.array([np.maximum(value, 0) for value in values_array], dtype=float)


def linear(values_array):
    return values_array


LAYERS_ACVT_FUNC = [relu, relu, linear]


def create_neural_net(layers_sizes, feature_qty):
    neural_net = np.zeros(len(layers_sizes), dtype=np.ndarray)
    neural_net[0] = np.random.uniform(-100, 100, (feature_qty + 1, layers_sizes[0]))

    for i in range(1, len(layers_sizes)):
        neural_net[i] = np.random.uniform(-1, 1, (layers_sizes[i - 1] + 1, layers_sizes[i]))
    return neural_net


def predict(neural_net, features):
    num_layers = neural_net.shape[0]
    result = np.array(np.append(features, 1))
    for i in range(num_layers):
        result = np.matmul(result, neural_net[i])
        result = LAYERS_ACVT_FUNC[i](result)
        result = np.append(result, 1)
    result = np.delete(result, -1)
    return result


def init_pop():
    ind = np.zeros(TAM_POP, dtype=np.ndarray)
    for i in range(TAM_POP):
        ind[i] = create_neural_net(LAYERS_SIZES, FEATURE_QTY)
    return ind


def avaliation(ind, x_data, y_data):
    fit = np.zeros(TAM_POP)
    # Temp array used to recive the predicted values from the neural network.
    predict_temp = np.zeros(len(x_data))
    min_mqe = 1000000
    for i in range(TAM_POP):  # For each neural network.
        for j in range(len(x_data)):  # For each value to be predicted.
            predict_temp[j] = predict(ind[i], x_data[j])
        # Calculates fit using the mean square error function of the predicted values and the correct values.
        b = mqe(predict_temp, y_data)
        if b < min_mqe:
            min_mqe = b
        a = 1 / (0.001 + b)
        fit[i] = a

    # print('Correct Data:\n', *y_data, '\n\n')
    # print('Data from Neural Network:\n', predict_temp)
    # print('fit = ', fit)
    print(min_mqe)
    return fit


def crossover(ind, fit):

    n_largest_fit_inds = heapq.nlargest(len(fit), range(len(fit)), fit.take)

    maxfit = fit[n_largest_fit_inds[0]]
    maxi = n_largest_fit_inds[0]
    delete_point = len(n_largest_fit_inds) // (2 / 3)

    for i in range(1, len(n_largest_fit_inds)):
        if i == maxi:
            continue
        curr_index = n_largest_fit_inds[i]

        if i >= delete_point:
            ind[curr_index] = create_neural_net(LAYERS_SIZES, FEATURE_QTY)
        else:
            # Using arithmetic mean.
            ind[curr_index] = (ind[curr_index] + ind[maxi]) / 2

        for k in range(len(ind[curr_index])):
            curr_matrix = ind[curr_index][k]
            i_shape, j_shape = ind[curr_index][k].shape
            for i_coord in range(i_shape):
                for j_coord in range(j_shape):
                    if np.random.random() < TAX_MUT:
                        curr_matrix[i_coord][j_coord] = ind[maxi][k][i_coord][j_coord] * \
                            2 * (np.random.random() + 0.1)


DATA_RANGE = (-50, 51, 1)
X_DATA = np.arange(*DATA_RANGE)
Y_DATA = math_function(X_DATA)

IND = init_pop()  # Start population with random values in a range.

while G < 201000:
    print('\n')
    print('AG Generation:', G, '\n')
    FIT = avaliation(IND, X_DATA, Y_DATA)
    crossover(IND, FIT)
    G = G + 1
