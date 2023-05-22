import perceptron as p
import numpy as np

# Training optical character recognition (OCR) network to classify integers
# from a seven-segment display.

OCR_network = p.MultiLayerPerceptron([7, 7, 10])
print('Training Neural Network as an Optical Character Recognizer...\n')
# The data for the ideal (full brightness on correct strip) number set
samples = [{'x': [1, 1, 1, 1, 1, 1, 0],            # 0     
            'y': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]},  
           {'x': [0, 1, 1, 0, 0, 0, 0],            # 1
            'y': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]},  
           {'x': [1, 1, 0, 1, 1, 0, 1],            # 2
            'y': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]},  
           {'x': [1, 1, 1, 1, 0, 0, 1],            # 3
            'y': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},  
           {'x': [0, 1, 1, 0, 0, 1, 1],            # 4
            'y': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]},  
           {'x': [1, 0, 1, 1, 0, 1, 1],            # 5
            'y': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]},  
           {'x': [1, 0, 1, 1, 1, 1, 1],            # 6
            'y': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]},  
           {'x': [1, 1, 1, 0, 0, 0, 0],            # 7
            'y': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]},  
           {'x': [1, 1, 1, 1, 1, 1, 1],            # 8
            'y': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]},  
           {'x': [1, 1, 1, 1, 0, 1, 1],            # 9
            'y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},]  

# Training epochs
num_epochs = 3000
for i in range(num_epochs):
    mse = 0.0
    for sample in samples:
        mse += OCR_network.bp(x=sample['x'], y=sample['y'])
    mse /= len(samples)
    if (i % 100) == 0:
        print(f'Error: {mse}')

print(f'\n Resulting weights after {num_epochs} epochs: ')
OCR_network.print_weights()

pattern = list(map(float, input("Input seven-segment pattern 'a b c d e f g': ").strip().split()))
result = np.argmax(OCR_network.run(pattern))
print(f'The OCR network recognized this number as {result}.')
    