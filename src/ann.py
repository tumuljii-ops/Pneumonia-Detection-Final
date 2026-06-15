import numpy as np

X_train = np.load("processed_data/X_train.npy")
X_test = np.load("processed_data/X_test.npy")

y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

#step 1


input_size = X_train.shape[1]

hidden1 = 128
hidden2 = 64
hidden3 = 32

output_size = 1

#step 2

np.random.seed(42)

W1 = np.random.randn(
    input_size,
    hidden1
) * 0.01

b1 = np.zeros((1, hidden1))


W2 = np.random.randn(
    hidden1,
    hidden2
) * 0.01

b2 = np.zeros((1, hidden2))


W3 = np.random.randn(
    hidden2,
    hidden3
) * 0.01

b3 = np.zeros((1, hidden3))


W4 = np.random.randn(
    hidden3,
    output_size
) * 0.01

b4 = np.zeros((1, output_size))

#relu
def relu(Z):

    return np.maximum(0, Z)

#sigmoid
def sigmoid(Z):

    return 1 / (1 + np.exp(-Z))

#forward propagation

def forward(X):

    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = relu(Z3)

    Z4 = np.dot(A3, W4) + b4
    A4 = sigmoid(Z4)

    return (
        Z1, A1,
        Z2, A2,
        Z3, A3,
        Z4, A4
    )
    
#training

Z1,A1,Z2,A2,Z3,A3,Z4,A4 = forward(
    X_train
)

print("A1:", A1.shape)
print("A2:", A2.shape)
print("A3:", A3.shape)
print("A4:", A4.shape)

print("\nFirst 5 Predictions:")
print(A4[:5])


def compute_loss(y_true, y_pred):

    y_true = y_true.reshape(-1,1)

    epsilon = 1e-8

    loss = -np.mean(
        y_true*np.log(y_pred + epsilon)
        +
        (1-y_true)*np.log(
            1-y_pred + epsilon
        )
    )

    return loss

loss = compute_loss(
    y_train,
    A4
)

print("\nInitial Loss:")
print(loss)

# ==========================================================
# RELU DERIVATIVE
# ==========================================================

def relu_derivative(Z):

    # Create array of zeros and ones
    # If Z > 0 => derivative = 1
    # If Z <= 0 => derivative = 0

    return (Z > 0).astype(np.float32)

# ==========================================================
# BACKPROPAGATION
# ==========================================================

def backward(
    X,
    y_true,

    Z1,A1,
    Z2,A2,
    Z3,A3,
    Z4,A4
):

    # Number of training samples

    m = X.shape[0]

    # Convert shape from:
    # (4172,)
    # to
    # (4172,1)

    y_true = y_true.reshape(-1,1)

    # ======================================================
    # OUTPUT LAYER
    # ======================================================

    # For sigmoid + BCE
    #
    # dZ4 = A4 - y
    #
    # Shape:
    # (4172,1)

    dZ4 = A4 - y_true

    # Gradient of W4
    #
    # A3.T = (32,4172)
    # dZ4  = (4172,1)
    #
    # Result:
    # (32,1)

    dW4 = (1/m) * np.dot(
        A3.T,
        dZ4
    )

    # Gradient of b4
    #
    # Shape:
    # (1,1)

    db4 = (1/m) * np.sum(
        dZ4,
        axis=0,
        keepdims=True
    )

    # ======================================================
    # HIDDEN LAYER 3
    # ======================================================

    # Move error backwards
    # dZ4 = (4172,1)
    # W4.T = (1,32)
    #
    # Result:
    # (4172,32)

    dA3 = np.dot(
        dZ4,
        W4.T
    )

    # Apply chain rule
    #
    # dZ = dA * ReLU'

    dZ3 = dA3 * relu_derivative(Z3)

    # Shape:
    # (64,4172) x (4172,32)
    # =>
    # (64,32)

    dW3 = (1/m) * np.dot(
        A2.T,
        dZ3
    )

    db3 = (1/m) * np.sum(
        dZ3,
        axis=0,
        keepdims=True
    )

    # ======================================================
    # HIDDEN LAYER 2
    # ======================================================

    dA2 = np.dot(
        dZ3,
        W3.T
    )

    dZ2 = dA2 * relu_derivative(Z2)

    dW2 = (1/m) * np.dot(
        A1.T,
        dZ2
    )

    db2 = (1/m) * np.sum(
        dZ2,
        axis=0,
        keepdims=True
    )

    # ======================================================
    # HIDDEN LAYER 1
    # ======================================================

    dA1 = np.dot(
        dZ2,
        W2.T
    )

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = (1/m) * np.dot(
        X.T,
        dZ1
    )

    db1 = (1/m) * np.sum(
        dZ1,
        axis=0,
        keepdims=True
    )

    # ======================================================
    # RETURN ALL GRADIENTS
    # ======================================================

    return (

    dW1,db1,

    dW2,db2,

    dW3,db3,

    dW4,db4
)
 
    
   # ==========================================================
# TEST BACKPROPAGATION
# ==========================================================

dW1,db1,\
dW2,db2,\
dW3,db3,\
dW4,db4 = backward(

    X_train,
    y_train,

    Z1,A1,
    Z2,A2,
    Z3,A3,
    Z4,A4
)

print("\nGradient Shapes\n")

print("dW1:", dW1.shape)
print("db1:", db1.shape)

print("dW2:", dW2.shape)
print("db2:", db2.shape)

print("dW3:", dW3.shape)
print("db3:", db3.shape)

print("dW4:", dW4.shape)
print("db4:", db4.shape)


# ==========================================================
# UPDATE PARAMETERS
# ==========================================================

def update_parameters(

    learning_rate,

    dW1,db1,
    dW2,db2,
    dW3,db3,
    dW4,db4
):

    global W1,b1
    global W2,b2
    global W3,b3
    global W4,b4

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4


# ==========================================================
# TRAINING LOOP
# ==========================================================

epochs = 100
learning_rate = 0.01

for epoch in range(epochs):

    # Forward Propagation

    Z1,A1,Z2,A2,Z3,A3,Z4,A4 = forward(
        X_train
    )

    # Loss

    loss = compute_loss(
        y_train,
        A4
    )

    # Backpropagation

    dW1,db1,\
    dW2,db2,\
    dW3,db3,\
    dW4,db4 = backward(

        X_train,
        y_train,

        Z1,A1,
        Z2,A2,
        Z3,A3,
        Z4,A4
    )

    # Gradient Descent

    update_parameters(

        learning_rate,

        dW1,db1,
        dW2,db2,
        dW3,db3,
        dW4,db4
    )

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | Loss = {loss:.6f}"
        )


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict(X):

    _,_,_,_,_,_,_,A4 = forward(X)

    predictions = (A4 >= 0.5).astype(int)

    return predictions


# ==========================================================
# TRAIN ACCURACY
# ==========================================================

train_predictions = predict(X_train)

train_accuracy = np.mean(
    train_predictions.flatten()
    ==
    y_train
)

print(
    "\nTrain Accuracy:",
    train_accuracy * 100
)


# ==========================================================
# TEST ACCURACY
# ==========================================================

test_predictions = predict(X_test)

test_accuracy = np.mean(
    test_predictions.flatten()
    ==
    y_test
)

print(
    "Test Accuracy:",
    test_accuracy * 100
)