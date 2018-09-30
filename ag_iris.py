import numpy as np
import pandas as pd


dataset_path = './dataset/iris.data'


def class_names_to_onehot(dataset):
    classes_to_onehot = {}

    for index, row in dataset.iterrows():
        if row['class'] not in classes_to_onehot:
            classes_to_onehot[row['class']] = 0

    num_of_classes = len(classes_to_onehot)
    index = 0
    base_onehot = [0] * num_of_classes
    for key, value in classes_to_onehot.items():
        temp_onehot = base_onehot.copy()
        temp_onehot[index] = 1
        classes_to_onehot[key] = temp_onehot
        index += 1

    return classes_to_onehot


def relu(values_array):
    return np.array([np.maximum(value, 0) for value in values_array], dtype=float)


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


def main():
    dataset = pd.read_csv(dataset_path, sep=',')
    classes_to_onehot = class_names_to_onehot(dataset)


if __name__ == '__main__':
    main()
