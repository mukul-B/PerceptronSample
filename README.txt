perceptron_test

1) for each sample, calculate activation by using numpy operation on X,w,b
2)update w and b if activation times label is less then 0
3) repeat 1 and 2 untill activation remain same for all samples or max_epoch is reached

perceptron_test

1) for each sample, calculate activation by using numpy operation on X,w,b
2) if activation and real label are of same sign , accuracy variable is incremented
3) return accuracy/ total sample

gradient_descent

1) get gradiant value at initial x value
2) get new x value using numpy operation on x and gradient value
3) repeat 1 and 2 but for new x , until gradient value is less then 0.0001
