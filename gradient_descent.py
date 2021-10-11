import numpy as np


def gradient_descent(gradiant, init_x, n):
    for i in range(50000):
        grad = gradiant(init_x)

        new_x = np.subtract(init_x, n * grad)
        if np.all(grad < 0.0001):
            break
        init_x = new_x

    return init_x
