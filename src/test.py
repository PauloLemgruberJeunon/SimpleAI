import neuralnet as nn


nf = nn.NeuralFactory()
nn = nf.create_neural_net(hidden_actv_fcn=nn.ActivationFcn(), final_actv_fcn=nn.ActivationFcn(),
                          layers_sizes=[1, 1], input_size=1)

for l in nn.layers:
    print(l.weight_mtr)

print('\n')
print(nn.predict(input_array=[1]))
