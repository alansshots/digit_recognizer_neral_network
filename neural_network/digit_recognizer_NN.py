import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and preprocess data
data = pd.read_csv('./train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def compute_loss(A2, Y):
    one_hot_Y = one_hot(Y)
    m = Y.shape[0]
    log_likelihood = -np.log(A2[one_hot_Y == 1])
    loss = np.sum(log_likelihood) / m
    return loss

loss_history = []

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            loss = compute_loss(A2, Y)
            loss_history.append(loss)
            print(f"Iteration {i}: Loss = {loss}, Accuracy = {accuracy}")
    return W1, b1, W2, b2

# Save and Load Functions
def save_nn(W1, b1, W2, b2, directory):
    np.savetxt(f'{directory}/W1.csv', W1, delimiter=',')
    np.savetxt(f'{directory}/b1.csv', b1, delimiter=',')
    np.savetxt(f'{directory}/W2.csv', W2, delimiter=',')
    np.savetxt(f'{directory}/b2.csv', b2, delimiter=',')

def load_nn(directory):
    W1 = np.loadtxt(f'{directory}/W1.csv', delimiter=',')
    b1 = np.loadtxt(f'{directory}/b1.csv', delimiter=',').reshape(-1, 1)
    W2 = np.loadtxt(f'{directory}/W2.csv', delimiter=',')
    b2 = np.loadtxt(f'{directory}/b2.csv', delimiter=',').reshape(-1, 1)
    return W1, b1, W2, b2

# Training the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1400)

# Save the model parameters
save_nn(W1, b1, W2, b2, './trained_data')

# Load the model parameters
W1, b1, W2, b2 = load_nn('./trained_data')

# Plot training loss
plt.plot(loss_history)
plt.xlabel('Iterations (per 100)')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.show()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    # Plot Img + Prediction and Label 
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.title(f"Prediction: {prediction[0]}, Label: {label}")
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)