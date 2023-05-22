import numpy as np


class Perceptron():
    """ A single neuron with sigmoid activation function."""

    def __init__(self, num_inputs, bias=1.0):
        self.weights = (np.random.rand(num_inputs + 1) * 2) - 1
        self.bias = bias

    def run(self, inputs: list):
        weighted_sum = np.dot(np.append(inputs, self.bias), self.weights)
        return self.sigmoid(weighted_sum)
    
    def sigmoid(self, weighted_sum: float):
        return 1/(1 + np.exp(-weighted_sum))

    def set_weights(self, weights: list):
        self.weights = np.array(weights)


class MultiLayerPerceptron():
    """ A multilayer perceptron class that uses the Perceptron class. """

    def __init__(self, layers: list, bias: float=1.0, eta: float=0.5):
        """ Layers is a list of integers. [num_inputs, num_neurons in layer 1, 
            num_neurons in layer 2, ..., num_outputs] """
        self.layers = np.array(layers, dtype=object)  
        self.bias = bias
        self.eta = eta

        network = []        # List of lists of Neurons
        values = []         # List of lists of output values
        d = []              # The list of lists of the error terms (lowercase 
                            # deltas) for backpropogation.
        for i, layer in enumerate(self.layers):
            init_list = [0.0 for num in range(layer)]
            values.append(init_list)
            d.append(init_list)
            network.append([])
            if i > 0:
                for num in range(layer):
                    network[i].append(Perceptron(num_inputs=self.layers[i-1], 
                                                 bias=self.bias))
        # Casting the list of lists to numpy arrays
        self.network = np.array([np.array(x) for x in network], dtype=object)
        self.values = np.array([np.array(x) for x in values], dtype=object)
        self.d = np.array([np.array(x) for x in d], dtype=object)
                    
    def set_weights(self, w_init: list):
        """ Set the weights with a list of lists for all layers but the input. """
        for i, _ in enumerate(w_init):
            for j, _ in enumerate(w_init[i]):
                self.network[i+1][j].set_weights(w_init[i][j])

    def print_weights(self):
        """ Visualize the weights at each layer. """
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                weights = self.network[i][j].weights
                print(f'Layer {i+1} Neuron {j}, {weights}')
        print()

    def run(self, x: list):
        """ Feed a sample x into the multi-layered perceptron. """
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]
    
    def bp(self, x: list, y: list):
        """ Run a single sample (x, y) pair through the backpropogation algo. """
        x = np.array(x, dtype=object)       # X is the feature input data
        y = np.array(y, dtype=object)       # Y is the label expected output

        # 1. Feed sample to the network
        ouputs = self.run(x)

        # 2. Calculate Mean Squared error
        error = (y - ouputs)
        MSE = sum(error ** 2)/self.layers[-1]

        # 3. Calculate the ouput error terms (k = kth neuron in the output layer)
        self.d[-1] = ouputs * (1 - ouputs) * (error)

        # 4. Calculate the hidden layer error terms by backpropogating through the
        #    hidden layers.
        for i, layer in reversed(list(enumerate(self.network))):
            # Skip the ouput layer and break when the input layer is reached
            if i == len(self.network) - 1:
                continue
            elif i == 0:
                break
            # Iterate through each neuron in each layer
            for h, neuron in enumerate(layer):
                fwd_error = 0.0
                # Iterate through the weights of the next layer (one layer ahead)
                for k in range(self.layers[i + 1]):
                    fwd_error += self.network[i + 1][k].weights[h] * self.d[i + 1][k]
                # Values were calculated and stored when the MLP was ran in step 1.
                o_h = self.values[i][h]
                self.d[i][h] = o_h * (1 - o_h) * fwd_error

        # 5 & 6. Calculate the deltas and update the weights (i = ith neuron in layer,
        #        j = jth input to that neuron)
        for l, layer in enumerate(self.network):
            # Skipping the input layer
            if not l:
                continue
            for i, neuron in enumerate(layer):
                for j in range(self.layers[l - 1] + 1):
                    # Handle the bias input
                    if j == self.layers[l-1]:
                        dW_ij = self.eta * self.d[l][i] * self.bias
                    else:
                        dW_ij = self.eta * self.d[l][i] * self.values[l-1][j]
                    neuron.weights[j] += dW_ij
        return MSE
