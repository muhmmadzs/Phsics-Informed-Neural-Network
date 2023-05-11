# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:25:06 2023

@author: muhamzs
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants for the pendulum
g = 9.81  # gravity
L1, L2 = 1.0, 1.0  # lengths of the pendulum arms
m1, m2 = 10.0, 10.0  # masses of the pendulum bobs

# Function to compute the derivatives of the state variables
def derivatives(y, t, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    theta1_dot = z1
    z1_dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2_dot = z2
    z2_dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1_dot, z1_dot, theta2_dot, z2_dot

# Generate training data using the true dynamics
t = np.linspace(0, 10, 1000)
y0 = [np.pi/2, 0, np.pi/2, 0]

states = odeint(derivatives, y0, t, args=(L1, L2, m1, m2))

# Prepare the data for LSTM
X = states[:-1, :]
Y = states[1:, :]

# Reshape the data for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))
Y = Y.reshape((Y.shape[0], Y.shape[1]))

class PhysicsInformedLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        theta1, z1, theta2, z2 = tf.split(inputs, num_or_size_splits=4, axis=1)
        theta1_dot = z1
        z1_dot = (m2*g*tf.sin(theta2)*tf.cos(theta1-theta2) - m2*tf.sin(theta1-theta2)*(L1*z1**2*tf.cos(theta1-theta2) + L2*z2**2) - (m1+m2)*g*tf.sin(theta1)) / L1 / (m1 + m2*tf.sin(theta1-theta2)**2)
        theta2_dot = z2
        z2_dot = (L1*(m1+m2)*z1**2*tf.sin(theta1-theta2) + (m1+m2)*g*tf.sin(theta1) + m2*L2*z2**2*tf.sin(theta1-theta2)*tf.cos(theta1-theta2) - m2*g*tf.sin(theta2)*tf.cos(theta1-theta2)) / L2 / (m1 + m2*tf.sin(theta1-theta2)**2)
        return tf.concat([theta1_dot, z1_dot, theta2_dot, z2_dot], axis=1)



# Define the LSTM model
model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=(None, 4)))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(4, activation='tanh', return_sequences=False))  
model.add(PhysicsInformedLayer())
model.add(Dense(4))

# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=opt, loss='mse')

# Train the model
model.fit(X, Y, epochs=200)





predictions = model.predict(X)


# Plot the true and predicted states
for i in range(4):
    plt.figure()
    plt.plot(t[:-1], states[:-1, i], label='True')
    plt.plot(t[:-1], predictions[:, i], label='Predicted')
    plt.legend()
    plt.show()


# Extract the predicted and true angles for the two pendulums
theta1_true = states[:-1, 0]
theta2_true = states[:-1, 2]
theta1_pred = predictions[:, 0]
theta2_pred = predictions[:, 2]

# Compute the (x, y) coordinates of the pendulums
x1_true = L1 * np.sin(theta1_true)
y1_true = -L1 * np.cos(theta1_true)
x2_true = x1_true + L2 * np.sin(theta2_true)
y2_true = y1_true - L2 * np.cos(theta2_true)

x1_pred = L1 * np.sin(theta1_pred)
y1_pred = -L1 * np.cos(theta1_pred)
x2_pred = x1_pred + L2 * np.sin(theta2_pred)
y2_pred = y1_pred - L2 * np.cos(theta2_pred)

fig, ax = plt.subplots()

# Initialize the pendulum lines that will be updated in the animation
line_true, = ax.plot([], [], 'o-', lw=2, color='blue', label='True')
line_pred, = ax.plot([], [], 'o-', lw=2, color='red', label='Predicted')

def init():
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.legend()
    return line_true, line_pred,

def update(i):
    thisx_true = [0, x1_true[i], x2_true[i]]
    thisy_true = [0, y1_true[i], y2_true[i]]
    line_true.set_data(thisx_true, thisy_true)
    
    thisx_pred = [0, x1_pred[i], x2_pred[i]]
    thisy_pred = [0, y1_pred[i], y2_pred[i]]
    line_pred.set_data(thisx_pred, thisy_pred)
    return line_true, line_pred,

ani = animation.FuncAnimation(fig, update, frames=range(len(predictions)),
                              init_func=init, blit=True)

plt.show()
