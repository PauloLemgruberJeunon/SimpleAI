import heapq
import csv
from random import randint as rint
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import mean_squared_error as mqe
np.set_printoptions(precision=3)

#Investigar 'Common Mistakes' https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
#https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

TAM_POP = 30
TAX_MUT = 0.01
LAYERS_SIZES = [15, 15, 3]
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


def init_pop():
    ind = np.zeros(TAM_POP, dtype=np.ndarray)
    for i in range(TAM_POP):
        ind[i] = create_neural_net(LAYERS_SIZES, FEATURE_QTY)
    return ind


def iris_avaliation(ind, x_data):
    fit = np.zeros(TAM_POP)
    # Temp array used to recive the predicted values from the neural network.
    predict_temp = np.zeros(len(x_data), dtype=np.ndarray)
    for i in range(TAM_POP):  # For each neural network.
        for j in range(len(x_data)):  # For each value to be predicted.
            predict_temp[j] = predict(ind[i], x_data[j][:4])
            
            #Contabilizando os erros.
            if(x_data[j][4] == 0 and (predict_temp[j][0] < 0.5 or predict_temp[j][1] > 0.5 or predict_temp[j][2] > 0.5)):
                fit[i] += 1 
            if(x_data[j][4] == 1 and (predict_temp[j][1] < 0.5 or predict_temp[j][0] > 0.5 or predict_temp[j][2] > 0.5)):
                fit[i] += 1
            if(x_data[j][4] == 2 and (predict_temp[j][2] < 0.5 or predict_temp[j][1] > 0.5 or predict_temp[j][0] > 0.5)):
                fit[i] += 1

            #print('Predict: ', predict_temp[j])
            #print('Dado correto: ', x_data[j][4])

        print('Geração: ', G)
        print('Erros: ', fit[i])
        fit[i] = 1/((fit[i]/len(x_data))+1)  #https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4667170
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

        a = rint(0, TAM_POP-1)
        b = rint(0, TAM_POP-1)
        if fit[a] > fit[b]:
            pai1 = a
        else:
            pai1 = b

        a = rint(0, TAM_POP-1)
        b = rint(0, TAM_POP-1)
        if fit[a] > fit[b]:
            pai2 = a
        else:
            pai2 = b

        crossover_index = [pai1, pai2, maxi]
        k = 0
        for j in range(len(LAYERS_SIZES)):
            rand = rint(0, len(LAYERS_SIZES)-1)
            ind[i][rand] = ind[crossover_index[k]][rand]
            rand = rint(0, len(LAYERS_SIZES)-1)
            ind[i][rand] = ind[i][rand] + np.random.uniform(-1, 1) * TAX_MUT
            k += 1
            if k == 3:
                k = 0

IND = init_pop()  # Start population with random values in a range.

with open("IrisDataSet/iris.data") as csv_file:
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
#plt.show()

plt.plot(DATA[0:50, 2:3], 'o')
plt.plot(DATA[50:100, 2:3], 'o')
plt.plot(DATA[100:150, 2:3], 'o')
plt.xlabel('petal length in cm')
plt.ylabel('petal width in cm')
plt.legend('012')
#plt.show()

# Média zero
DATA[:,0] = DATA[:,0] - np.mean(DATA[:,0])
DATA[:,1] = DATA[:,1] - np.mean(DATA[:,1])
DATA[:,2] = DATA[:,2] - np.mean(DATA[:,2])
DATA[:,3] = DATA[:,3] - np.mean(DATA[:,3])

# Variância 1
DATA[:,0] = DATA[:,0]/np.std(DATA[:,0])
DATA[:,1] = DATA[:,1]/np.std(DATA[:,1])
DATA[:,2] = DATA[:,2]/np.std(DATA[:,2])
DATA[:,3] = DATA[:,3]/np.std(DATA[:,3])

while G < 201:
    print('\n')
    print('AG Generation:', G, '\n')
    FIT = iris_avaliation(IND, DATA)
    iris_crossover(IND, FIT)
    G = G + 1
