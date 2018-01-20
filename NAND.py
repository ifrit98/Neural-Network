# dependencies
import numpy as np

# activation function
def sigmoid(z, derivative=False):
    if (derivative == True):
        return (z * (1 - z))
    return 1 / (1 + np.exp(-z))

# input data, 3rd column for bias units
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output training data for NAND
y = np.array([[0],
              [1],
              [1],
              [1]])

# generate random seed so it is the same each time
np.random.seed(1)

# Theta matrices

# 3x4 matrix of weights
Theta0 = 2 * np.random.random((3, 4)) - 1
# 4x1 matrix of weights
Theta1 = 2 * np.random.random((4, 1)) - 1

for j in range(100000):
    # Forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, Theta0))
    layer2 = sigmoid(np.dot(layer1, Theta1))

    # Back propagation
    l2_error = y - layer2

    if (j % 25000) == 0:  # Only print the error every 25000 steps
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(layer2, derivative=True)
    l1_error = l2_delta.dot(Theta1.T)
    l1_delta = l1_error * sigmoid(layer1, derivative=True)

    # Update weights
    Theta1 += layer1.T.dot(l2_delta)
    Theta0 += layer0.T.dot(l1_delta)

print("Output: ")
print(layer2)
