# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Deep Learning, Andrew Ng
# Video 26 - Vectorizing Across Multiple Training
#            Examples

import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-1. * z))

# single observation,Multiple features,
# single layer, single Perceptron ("circle")
x = np.array([[1],[2],[3]])

w = np.array([0.10 ,0.15, 0.30]) # Randomly assigned at start, then adjusted by
b = 0.2                        # the forward/backward propogation steps.

z = np.dot(w,x) + b
a = sigmoid(z)
print(a)

# Single Observation , multiple features (x1, x2, x3),
# single layers, multiple perceptron ("circle") (in this
# case: 4 hidden layder perceptrons)

x = np.array([[1],[2],[3]])

w_l1_p1 = np.array([0.10 ,0.15, 0.30])
b_l1_p1 = 0.2
w_l1_p2 = np.array([0.20 ,0.25, 0.32])
b_l1_p2 = 0.4
w_l1_p3 = np.array([0.21 ,0.12, 0.32])
b_l1_p3 = 0.4
w_l1_p4 = np.array([0.51 ,0.15, 0.53])
b_l1_p4 = 0.4

z_l1_p1 = np.dot(w_l1_p1,x) + b_l1_p1
a_l1_p1 = sigmoid(z_l1_p1)

z_l1_p2 = np.dot(w_l1_p2,x) + b_l1_p2
a_l1_p2 = sigmoid(z_l1_p2)

z_l1_p3 = np.dot(w_l1_p3,x) + b_l1_p3
a_l1_p3 = sigmoid(z_l1_p3)

z_l1_p4 = np.dot(w_l1_p4,x) + b_l1_p4
a_l1_p4 = sigmoid(z_l1_p4) 
print(a_l1_p1)
print(a_l1_p2)
print(a_l1_p3)
print(a_l1_p4)


# Now do the same thing, except vecorize it.
x = np.array([[1],[2],[3]])


# Column 1: Weights of x1, x2, x3 features (respectively) to perceptron 1
# Column 2: Weights of x2, x2, x3 features (respectively) to perceptron 2
# and so on...
# Size does not depend, then, on 'm' (the # of observations of x)
# W will always be #inputs (features) x #outputs (perceptrons)
# or nXp (for the current layer)
W_l1 = np.array([[ 0.10,  0.20,  0.21,  0.51 ],
                 [ 0.15,  0.25,  0.12,  0.15 ],
                 [ 0.30,  0.32,  0.32,  0.53 ]])
             

b_l1 = np.array([[0.2], [0.4], [0.4], [0.4]])

z_l1 = np.dot(W_l1.T,x) + b_l1
a_l1 = sigmoid(z_l1)
print(a_l1)
# and for reference, it's the exact same thing
print("")
print(a_l1_p1)
print(a_l1_p2)
print(a_l1_p3)
print(a_l1_p4)

# Now we can add layer 2, which is only
# One perceptron, weighting the 4 results from the previous layer
# So size input X output = 4x1 (input x output)
w_l2 = np.array([[0.1], 
                 [0.2], 
                 [0.3], 
                 [0.4]])
b_l2 = 0.3

z_l2 = np.dot(w_l2.T,a_l1) + b_l2
a_l2 = sigmoid(z_l2)
print(a_l2)

# Now we've done the forward propogation for one
# observation of x. We have m observations to calculate.
# We could do a for-loop from 1-to-m. Or we can again
# vectorize.

# Our matrix X will be all of our m observations of x. (we'll use 10)
# Each still with 3 features. We want each observation in a column.
# So the matrix will be n-rows (features) by m-columns (observations)
# or in this case 3x10
#               X(1) X(2)       .    .    .                           X(m)
X = np.array([[ 1. , -3. ,  0.9, -0.7,  0.3,  1.6, -1.2,  0.1,  0.3,  0.5], #X1
              [ 2. , -1.1, -1.1,  0.3, -1.5, -0.7,  0.2,  0.1,  0.1,  0.5], #x2
              [ 3. ,  1.1,  0.4,  0.8, -1.9, -0.8,  0.4,  0. ,  0.8,  0.9]])#X3

W_l1 = np.array([[ 0.10,  0.20,  0.21,  0.51 ],
                 [ 0.15,  0.25,  0.12,  0.15 ],
                 [ 0.30,  0.32,  0.32,  0.53 ]])

# Remember, as stated above, w is #inputs (perceptrons) x #outputs (x-es)
# so 3x4

# b is still one bias value per perceptron, in that layer
B_l1 = np.array([[0.2], [0.4], [0.4], [0.4]])

# We need dimensions to line up, so must transpose w
Z_l1 = np.dot(W_l1.T, X) + B_l1

A_l1 = sigmoid(Z_l1)
'''array([[ 0.81757448,  0.51624428,  0.56094545,  0.60228618,  0.36239135,
         0.50374993,  0.55724785,  0.55601389,  0.61892785,  0.64451155],
       [ 0.84553473,  0.565865  ,  0.60944978,  0.64908052,  0.40974975,
         0.5535439 ,  0.60587367,  0.60467908,  0.66485373,  0.68890392],
       [ 0.84553473,  0.565865  ,  0.60944978,  0.64908052,  0.40974975,
         0.5535439 ,  0.60587367,  0.60467908,  0.66485373,  0.68890392],
       [ 0.84553473,  0.565865  ,  0.60944978,  0.64908052,  0.40974975,
         0.5535439 ,  0.60587367,  0.60467908,  0.66485373,  0.68890392]])
'''
# So we can see the same numbers in for the first observation as above.
print(A_l1[:,0].reshape(4,1))
print("")
print(a_l1)


# And now we finish off with layer 2

# size of W = in x out = 4x1, same as before
W_l2 = np.array([[0.1], 
                 [0.2], 
                 [0.3], 
                 [0.4]])
b_l2 = 0.3

Z_l2 = np.dot(W_l2.T, A_l1) + b_l2
A_l2 = sigmoid(Z_l2)

print(A_l2)
print("")
print(A_l2[:,0])
print("")
print(a_l2)


