import heapq
import csv
from random import randint as rint
import matplotlib.pyplot as plt
import numpy as np
from random import randint


np.set_printoptions(precision=3)

# Investigar 'Common Mistakes' https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
# https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

TAM_POP = 15
TAX_MUT = 0.3
CROSS_PROB = 0.2
LAYERS_SIZES = [10, 10, 3]
FEATURE_QTY = 4
MAXX = 1
G = 1


def relu(values_array):
    return np.array([np.maximum(value, 0) for value in values_array], dtype=float)


def sigmoid(values_array):
    return 1 / (1 + np.exp(-values_array))


def softmax(values_array):
    # print(values_array)
    # quit()
    score_mat_exp = np.exp(np.asarray(values_array))
    return score_mat_exp / score_mat_exp.sum(0)


LAYERS_ACVT_FUNC = [relu, relu, softmax]


def create_neural_net(layers_sizes, feature_qty):
    neural_net = np.zeros(len(layers_sizes), dtype=np.ndarray)
    neural_net[0] = np.random.uniform(-2, 2, (feature_qty + 1, layers_sizes[0]))

    for i in range(1, len(layers_sizes)):
        neural_net[i] = np.random.uniform(-2, 2, (layers_sizes[i - 1] + 1, layers_sizes[i]))
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


def iris_avaliation(ind, x_data):
    fit = np.zeros(TAM_POP)

    # Temp array used to recive the predicted values from the neural network.
    predict_temp = np.zeros(len(x_data), dtype=np.ndarray)
    acc = np.zeros(TAM_POP, dtype=float)
    for i in range(TAM_POP):  # For each neural network.
        for j in range(len(x_data)):  # For each value to be predicted.
            predict_temp[j] = predict(ind[i], x_data[j][:4])

            # Contabilizando os erros.
            # print(np.abs(predict_temp[j][0] - 1) + predict_temp[j][1] + predict_temp[j][2])
            # quit()
            if(x_data[j][4] == 0):
                fit[i] += np.abs(predict_temp[j][0] - 1) + predict_temp[j][1] + predict_temp[j][2]
            if(x_data[j][4] == 1):
                fit[i] += np.abs(predict_temp[j][1] - 1) + predict_temp[j][0] + predict_temp[j][2]
            if(x_data[j][4] == 2):
                fit[i] += np.abs(predict_temp[j][2] - 1) + predict_temp[j][0] + predict_temp[j][1]

            if x_data[j][4] == np.argmax(predict_temp[j]):
                acc[i] += 1

        # print(fit[i])
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4667170
        fit[i] = 1 / (fit[i] + 1)
        acc[i] = acc[i] / len(x_data)

    return fit, acc


def iris_crossover(ind, fit, acc):

    ordered_indices = heapq.nlargest(len(fit), range(len(fit)), fit.take)
    maxi = ordered_indices[0]

    for i in range(int(TAM_POP * 0.6), TAM_POP):
        ind[ordered_indices[i]] = create_neural_net(LAYERS_SIZES, FEATURE_QTY)

    print('fit = ', fit[maxi])
    print('acc = ', acc[maxi])

    # for i in range(len(fit)):
    #     if i == maxi:
    #         continue
    #
    #     for j in range(len(LAYERS_SIZES)):
    #         if CROSS_PROB < np.random.uniform(0, 1):
    #             ind[i][j] = (ind[maxi][j] + ind[i][j]) / 2
    #         # for k in range(ind[i][j].shape[0]):
    #         #     for l in range(ind[i][j].shape[1]):
    #         #         ind[i][j][k][l] *= np.random.uniform(0.1, 2) * randint(-1, 1)


IND = init_pop()  # Start population with random values in a range.

with open("./dataset/iris.data") as csv_file:
    CSV_READER = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    DATA = []
    for row in CSV_READER:
        DATA.append(row)
    DATA.pop()
DATA = np.array(DATA, dtype=np.float64)

plt.plot(DATA[0:50, 0:1], 'o')
plt.plot(DATA[50:100, 0:1], 'o')
plt.plot(DATA[100:150, 0:1], 'o')
plt.xlabel('sepal length in cm')
plt.ylabel('sepal width in cm')
plt.legend('012')
# plt.show()

plt.plot(DATA[0:50, 2:3], 'o')
plt.plot(DATA[50:100, 2:3], 'o')
plt.plot(DATA[100:150, 2:3], 'o')
plt.xlabel('petal length in cm')
plt.ylabel('petal width in cm')
plt.legend('012')
# plt.show()

# Média zero
DATA[:, 0] = DATA[:, 0] - np.mean(DATA[:, 0])
DATA[:, 1] = DATA[:, 1] - np.mean(DATA[:, 1])
DATA[:, 2] = DATA[:, 2] - np.mean(DATA[:, 2])
DATA[:, 3] = DATA[:, 3] - np.mean(DATA[:, 3])

# Variância 1
DATA[:, 0] = DATA[:, 0] / np.std(DATA[:, 0])
DATA[:, 1] = DATA[:, 1] / np.std(DATA[:, 1])
DATA[:, 2] = DATA[:, 2] / np.std(DATA[:, 2])
DATA[:, 3] = DATA[:, 3] / np.std(DATA[:, 3])

while G < 201:
    print('\n')
    print('AG Generation:', G, '\n')
    FIT, ACC = iris_avaliation(IND, DATA)
    iris_crossover(IND, FIT, ACC)
    G = G + 1
