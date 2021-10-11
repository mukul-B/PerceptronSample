import numpy as np


def perceptron_train(X, Y):
    w = np.array([0 for i in X[0]])
    b = 0
    max_epock=10000
    while True:
        epic_without_update = 0
        for i in range(len(Y)):
            activation = np.multiply(X[i], w)
            activation = np.sum(activation) + b
            if activation * Y[i] <= 0:
                w = np.add(w, np.multiply(X[i], Y[i]))
                b = b + Y[i]
                epic_without_update = epic_without_update + 1
        max_epock=max_epock-1
        if epic_without_update == 0:
            break
        if max_epock==0:
            print("data not linearly convergerable. max epoch limit has reached")
            return np.array([0 for i in X[0]]),0

    return w, b


def perceptron_test(X, Y, w, b):
    #print(X, Y, w, b)
    accuracy = 0
    sample_size=len(Y)
    for i in range(sample_size):
        activation = np.multiply(X[i], w)
        activation = np.sum(activation) + b
        #print(activation)
        y_pred = -1 if activation <= 0 else 1
        if y_pred == Y[i]:
            accuracy = accuracy + 1

    return accuracy/float(sample_size)
