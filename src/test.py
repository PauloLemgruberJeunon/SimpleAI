import neuralnet as nn


nf = nn.NeuralFactory()
nn = nf.create_neural_net(hidden_actv_fcn=nn.ActivationFcn(), final_actv_fcn=nn.ActivationFcn(),
                          layers_sizes=[1, 1], input_size=2)

for l in nn.layers:
    print(l.weight_mtr)

# print('\n')
# print(nn.predict(input_array=[1, 2]))
# print()

x_array = [[1, 1], [0, 1], [1, 0], [0, 0]]
y_array = [[0], [1], [1], [0]]

nn.fit(x_array=x_array, y_array=y_array, epochs=1000, n=2)
