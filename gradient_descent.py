import numpy as np
def gradient_descent(gradiant, init_x, n):
    for i in range(52):
        grad = gradiant(init_x)
        new_x = np.subtract(init_x, n * grad)
        #print(i, new_x, init_x, grad)
        if grad.all() == 0:
            break
        init_x = new_x

    return init_x
