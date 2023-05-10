# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:10:56 2023
Physics Informed Machine learning (Harmonic Oscillator Example)
@author: muhamzs
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# Constants
omega = 5.0

# Define the custom layer
class PhysicsInformedLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x, v = tf.split(inputs, num_or_size_splits=2, axis=1)
        dx_dt = v
        dv_dt = -omega**2 * x
        return tf.concat([dx_dt, dv_dt], axis=1)

# Define the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 2)),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(2, return_sequences=False),
    PhysicsInformedLayer()
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Define the system of differential equations
def harmonic_oscillator(t, y):
    x, v = y
    dx_dt = v
    dv_dt = -omega**2 * x
    return [dx_dt, dv_dt]

# Use solve_ivp to generate the training data
initial_conditions = [1.0, 0.0]  # example initial conditions
sol = solve_ivp(harmonic_oscillator, [0, 10], initial_conditions, t_eval=np.linspace(0, 10, 1000))

# Prepare the training data
X_train = []
Y_train = []

for i in range(len(sol.t) - 10):
    X_train.append(sol.y[:, i:i+10].T)
    Y_train.append(sol.y[:, i+10])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Train the model
model.fit(X_train, Y_train, epochs=500)

# Predict the future states of the system
initial_conditions_test = sol.y[:, :10].T  # Use the first 10 states from the true solution as initial conditions

predictions = []

for _ in range(100):  # Predict 100 steps into the future
    prediction = model.predict(initial_conditions_test.reshape(1, 10, 2))
    predictions.append(prediction[0])

    # Update the initial conditions by removing the first state and adding the predicted state
    initial_conditions_test = np.roll(initial_conditions_test, -1, axis=0)
    initial_conditions_test[-1, :] = prediction

predictions = np.array(predictions)

# Compare the predicted states with the true states
true_states = sol.y[:, 10:110].T  # The true states corresponding to the predicted states

# Plot the predicted states and the true states
time = np.arange(len(predictions))

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time, predictions[:, 0], label='Predicted')
plt.plot(time, true_states[:, 0], label='True')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, predictions[:, 1], label='Predicted')
plt.plot(time, true_states[:, 1], label='True')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

plt.tight_layout()
plt.show()



# Animation of the Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Set the limits of the plots
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

# Create the lines for the true and predicted data
line_true, = ax1.plot([], [], '-', lw=2, alpha=0.5)
line_pred, = ax2.plot([], [], '-', lw=2, alpha=0.5)
point_true, = ax1.plot([], [], 'o', lw=2)
point_pred, = ax2.plot([], [], 'o', lw=2)

# Set the titles of the plots
ax1.set_title('True')
ax2.set_title('Predicted')

# Variables to store the positions and velocities
x_true_data, v_true_data = [], []
x_pred_data, v_pred_data = [], []

# Initialization function
def init():
    line_true.set_data([], [])
    line_pred.set_data([], [])
    point_true.set_data([], [])
    point_pred.set_data([], [])
    return line_true, line_pred, point_true, point_pred,

# Animation function
def animate(i):
    x_true = true_states[i, 0]
    v_true = true_states[i, 1]
    x_pred = predictions[i, 0]
    v_pred = predictions[i, 1]

    x_true_data.append(x_true)
    v_true_data.append(v_true)
    x_pred_data.append(x_pred)
    v_pred_data.append(v_pred)

    line_true.set_data(x_true_data, v_true_data)
    line_pred.set_data(x_pred_data, v_pred_data)
    point_true.set_data(x_true, v_true)
    point_pred.set_data(x_pred, v_pred)
    return line_true, line_pred, point_true, point_pred,

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=100, blit=True)

# Display the animation
plt.show()
