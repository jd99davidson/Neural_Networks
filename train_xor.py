import perceptron as p

# Training the MLP to find the weights needed to operate as an XOR gate
xor_network = p.MultiLayerPerceptron([2, 2, 1])
print('Training Neural Network as an XOR Gate...\n')
# These are the training epochs (we are doing 3000 epochs)
num_epochs = 3000
for i in range(num_epochs):
    mse = 0.0
    mse += xor_network.bp(x=[0, 0], y=[0])
    mse += xor_network.bp(x=[0, 1], y=[1])
    mse += xor_network.bp(x=[1, 0], y=[1])
    mse += xor_network.bp(x=[1, 1], y=[0])
    mse /= 4
    if (i % 100) == 0:
        print(f'Error: {mse}')

print(f'\n Resulting weights after {num_epochs} epochs: ')
xor_network.print_weights()

print('\nXOR Gate:')
print(f'0 0 = {xor_network.run([0, 0])}')
print(f'0 1 = {xor_network.run([0, 1])}')
print(f'1 0 = {xor_network.run([1, 0])}')
print(f'1 1 = {xor_network.run([1, 1])}')