import perceptron as p

# Logical OR gate
or_neuron = p.Perceptron(num_inputs=2)
or_neuron.set_weights([15, 15, -10])

print('OR Gate:')
print(f'0 0 = {or_neuron.run([0, 0])}')
print(f'0 1 = {or_neuron.run([0, 1])}')
print(f'1 0 = {or_neuron.run([1, 0])}')
print(f'1 1 = {or_neuron.run([1, 1])}')

# Logical AND gate
and_nueron = p.Perceptron(num_inputs=2)
and_nueron.set_weights([10, 10, -15])

print('\nAND Gate:')
print(f'0 0 = {and_nueron.run([0, 0])}')
print(f'0 1 = {and_nueron.run([0, 1])}')
print(f'1 0 = {and_nueron.run([1, 0])}')
print(f'1 1 = {and_nueron.run([1, 1])}')

# Logical XOR gate
xor_network = p.MultiLayerPerceptron(layers=[2, 2, 1])
# Each neuron gets three weights (input1, input2, and bias). The structure of 
# 'weights' indicates which neuron at which layer gets the assigned weight
weights = [[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]]
xor_network.set_weights(weights)

print('\nXOR Gate:')
print(f'0 0 = {xor_network.run([0, 0])}')
print(f'0 1 = {xor_network.run([0, 1])}')
print(f'1 0 = {xor_network.run([1, 0])}')
print(f'1 1 = {xor_network.run([1, 1])}')
