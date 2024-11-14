import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Formula y=mx+b
m = 2
b = 0.5
x = np.linspace(0, 4, 100)
y = m * x + b + np.random.randn(*x.shape) + 0.25

plt.scatter(x, y)
plt.show()  # Shows the plot when running py-file on PyCharm


# the model class
class Model:
    def __init__(self):  # Unlike regular initializations, 0.0 may result in dead results
        self.weight = tf.Variable(10.0)
        self.bias = tf.Variable(10.0)

    def __call__(self, x):
        return self.weight * x + self.bias

# loss calculation function
def calculate_loss(y_actual, y_output):
    return tf.reduce_mean(tf.square(y_actual - y_output))

# training function
def train(model, x, y, learning_rate):
    with tf.GradientTape() as gt:
        y_output = model(x)
        loss = calculate_loss(y, y_output)

    new_weight, new_bias = gt.gradient(loss, [model.weight, model.bias])
    model.weight.assign_sub(learning_rate * new_weight)
    model.bias.assign_sub(learning_rate * new_bias)


# The training loop
model = Model()
epochs = 100
learning_rate = 0.15

for epoch in range(epochs):
    y_output = model(x)
    loss = calculate_loss(y, y_output)
    print(f"Epoch: {epoch}, Loss: {loss.numpy()}")
    train(model, x, y, learning_rate)


# Testing & Evaluation
print(f"Weight: {model.weight.numpy()}")
print(f"Bias: {model.bias.numpy()}")

# Testing with the new data
new_x = np.linspace(0, 4, 50)
new_y = model.weight.numpy() * new_x + model.bias.numpy()
plt.scatter(new_x, new_y)
plt.scatter(x, y)
plt.show()