import heapq
import csv
from random import randint as rint
import numpy as np
from sklearn.metrics import mean_squared_error as mqe
np.set_printoptions(precision=3)


TAM_POP = 30
TAX_MUT = 0.05
LAYERS_SIZES = [8, 8, 3]
FEATURE_QTY = 4
MAXX = 1
G = 1


def math_function(values_array):
    return [(x ** 2) + 5 for x in values_array]


def relu(values_array):
    return np.array([np.maximum(value, 0) for value in values_array], dtype=float)


def linear(values_array):
    return values_array


def sigmoid(values_array):
    return 1 / (1 + np.exp(-values_array))


def softmax(values_array):
    score_mat_exp = np.exp(np.asarray(values_array))
    return score_mat_exp / score_mat_exp.sum(0)

LAYERS_ACVT_FUNC = [relu, relu, softmax]

def create_neural_net(layers_sizes, feature_qty):
    neural_net = np.zeros(len(layers_sizes), dtype=np.ndarray)
    neural_net[0] = np.random.uniform(-1, 1, (feature_qty + 1, layers_sizes[0]))

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


def iris_create_neural_net(layers_sizes, feature_qty):
    neural_net = np.zeros(len(layers_sizes), dtype=np.ndarray)
    neural_net[0] = np.random.uniform(-1, 1, (feature_qty, layers_sizes[0]))

    for i in range(1, len(layers_sizes)):
        neural_net[i] = np.random.uniform(-1, 1, (layers_sizes[i - 1], layers_sizes[i]))
    return neural_net


def iris_predict(neural_net, features):
    num_layers = neural_net.shape[0]
    result = np.array(features)
    for i in range(num_layers):
        result = np.matmul(result, neural_net[i])
        bias = np.random.uniform(-1, 1, result.shape)
        result = result + bias
        result = LAYERS_ACVT_FUNC[i](result)
    return result


def init_pop():
    ind = np.zeros(TAM_POP, dtype=np.ndarray)
    for i in range(TAM_POP):
        ind[i] = iris_create_neural_net(LAYERS_SIZES, FEATURE_QTY)
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


def iris_avaliation(ind, x_data):
    fit = np.zeros(TAM_POP)
    # Temp array used to recive the predicted values from the neural network.
    predict_temp = np.zeros(len(x_data), dtype=np.ndarray)
    for i in range(TAM_POP):  # For each neural network.
        for j in range(len(x_data)):  # For each value to be predicted.
            predict_temp[j] = iris_predict(ind[i], x_data[j][:4])
            
            if(x_data[j][4] == 0 and (predict_temp[j][0] < 0.5 or predict_temp[j][1] > 0.5 or predict_temp[j][2] > 0.5)):
                fit[i] += 1
            if(x_data[j][4] == 1 and (predict_temp[j][1] < 0.5 or predict_temp[j][0] > 0.5 or predict_temp[j][2] > 0.5)):
                fit[i] += 1
            if(x_data[j][4] == 2 and (predict_temp[j][2] < 0.5 or predict_temp[j][1] > 0.5 or predict_temp[j][0] > 0.5)):
                fit[i] += 1

            #if G==20:
                #print('Predict: ',predict_temp[j])
                #print('Dado correto: ',x_data[j][4])

        print('Geração: ', G)
        print('Erros: ', fit[i])
        fit[i] = 1/((fit[i]/len(x_data))+1) 
        print('Fit: ', fit[i])

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


def iris_crossover(ind, fit):
    maxi = 0
    for i in range(len(fit)):
        if fit[i] > fit[maxi]:
            maxi = i

    for i in range(len(fit)):
        if i == maxi:
            continue

        a = int((np.random.rand() % TAM_POP) + 1)
        b = int((np.random.rand() % TAM_POP) + 1)
        if fit[a] > fit[b]:
            pai1 = a
        else:
            pai1 = b

        a = int((np.random.rand() % TAM_POP) + 1)
        b = int((np.random.rand() % TAM_POP) + 1)
        if fit[a] > fit[b]:
            pai2 = a
        else:
            pai2 = b

        rand = rint(0, len(LAYERS_SIZES)-1)
        ind[i][rand] = ind[pai1][rand]
        rand = rint(0, len(LAYERS_SIZES)-1)
        ind[i][rand] = ind[maxi][rand]
        rand = rint(0, len(LAYERS_SIZES)-1)
        ind[i][rand] = ind[pai2][rand]


IND = init_pop()  # Start population with random values in a range.

with open("IrisDataSet/iris.data") as csv_file:
    CSV_READER = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    DATA = []
    for row in CSV_READER:
        DATA.append(row)
    DATA.pop()
DATA = np.array(DATA, dtype=np.float64)

while G < 201:
    print('\n')
    print('AG Generation:', G, '\n')
    FIT = iris_avaliation(IND, DATA)
    iris_crossover(IND, FIT)
    G = G + 1
