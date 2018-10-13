import neuralnet as nn
import numpy as np
import csv

nf = nn.NeuralFactory()
nn = nf.create(hidden_actv_fcn=nn.ActivationFcn(), final_actv_fcn=nn.ActivationFcn(),
               layers_sizes=[6, 3], input_size=4)

with open("../dataset/iris.data") as csv_file:
    CSV_READER = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    DATA = []
    for row in CSV_READER:
        DATA.append(row)
    DATA.pop()
DATA = np.array(DATA, dtype=np.float64)

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

nn.fit(x_array=DATA[:, :4], y_array=DATA[:, 4:], epochs=1000, n=0.1)
