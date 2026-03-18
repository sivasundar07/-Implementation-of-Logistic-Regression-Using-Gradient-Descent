# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries (numpy, matplotlib) and define dataset X and Y.

2.Initialize parameters w, b, learning rate, and number of epochs.

3.Apply the \sigma(z)=\frac{1}{1+e^{-z}} function to compute predicted probabilities.

4.Use gradient descent to calculate gradients (dw, db) and update w and b.

5.Print the final weight and bias, then plot the logistic regression curve with the data. 

## Program:
```
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([0,1,2,3,4,5,6,7,8,9])
Y = np.array([0,0,0,0,0,1,1,1,1,1])

# Initialize parameters
w = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(epochs):

    # Linear model
    z = w * X + b

    # Prediction
    y_pred = sigmoid(z)

    # Gradients
    dw = (1/n) * np.sum((y_pred - Y) * X)
    db = (1/n) * np.sum(y_pred - Y)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight:", w)
print("Bias:", b)

# Predictions
z = w * X + b
prob = sigmoid(z)

plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, prob, color="red", label="Logistic Curve")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.show()
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Siva Sundar P
RegisterNumber: 25011320 
*/
```

## Output:
<img width="442" height="295" alt="image" src="https://github.com/user-attachments/assets/7f45ea1c-8545-4620-9996-f603b5cc856f" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

