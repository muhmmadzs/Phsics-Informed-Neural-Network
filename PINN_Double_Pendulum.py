# -*- coding: utf-8 -*-
"""
Physics-Informed Neural Network (PINN) for Double Pendulum System
This implementation uses automatic differentiation to enforce physics constraints
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DoublePendulumPINN:
    def __init__(self, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
        """
        Initialize the PINN for double pendulum
        
        Parameters:
        L1, L2: lengths of pendulum arms
        m1, m2: masses of pendulum bobs
        g: gravitational acceleration
        """
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        
        # Build the neural network
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network architecture"""
        model = keras.Sequential([
            layers.Dense(64, activation='tanh', input_shape=(1,)),
            layers.Dense(64, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(64, activation='tanh'),
            layers.Dense(4)  # Output: [theta1, theta1_dot, theta2, theta2_dot]
        ])
        return model
    
    def predict_state(self, t):
        """Predict the state variables at time t"""
        t_tensor = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float32)
        return self.model(t_tensor)
    
    def compute_derivatives(self, t):
        """Compute time derivatives using automatic differentiation"""
        with tf.GradientTape() as tape:
            t_tensor = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float32)
            state = self.model(t_tensor)
        
        # Compute first derivatives
        state_dot = tape.gradient(state, t_tensor)
        
        # Compute second derivatives
        with tf.GradientTape() as tape2:
            t_tensor = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float32)
            state = self.model(t_tensor)
            state_dot = tape2.gradient(state, t_tensor)
        
        state_ddot = tape2.gradient(state_dot, t_tensor)
        
        return state, state_dot, state_ddot
    
    def physics_loss(self, t):
        """Compute the physics-informed loss based on the equations of motion"""
        state, state_dot, state_ddot = self.compute_derivatives(t)
        
        # Extract state variables
        theta1 = state[:, 0]
        theta1_dot = state[:, 1]
        theta2 = state[:, 2]
        theta2_dot = state[:, 3]
        
        # Extract second derivatives
        theta1_ddot = state_ddot[:, 0]
        theta2_ddot = state_ddot[:, 2]
        
        # Compute trigonometric functions
        cos_diff = tf.cos(theta1 - theta2)
        sin_diff = tf.sin(theta1 - theta2)
        
        # Equations of motion for double pendulum
        # First pendulum equation
        term1_1 = self.m2 * self.g * tf.sin(theta2) * cos_diff
        term1_2 = self.m2 * sin_diff * (self.L1 * theta1_dot**2 * cos_diff + self.L2 * theta2_dot**2)
        term1_3 = (self.m1 + self.m2) * self.g * tf.sin(theta1)
        denominator1 = self.L1 * (self.m1 + self.m2 * sin_diff**2)
        
        residual1 = theta1_ddot - (term1_1 - term1_2 - term1_3) / denominator1
        
        # Second pendulum equation
        term2_1 = (self.m1 + self.m2) * (self.L1 * theta1_dot**2 * sin_diff - self.g * tf.sin(theta2) + self.g * tf.sin(theta1) * cos_diff)
        term2_2 = self.m2 * self.L2 * theta2_dot**2 * sin_diff * cos_diff
        denominator2 = self.L2 * (self.m1 + self.m2 * sin_diff**2)
        
        residual2 = theta2_ddot - (term2_1 + term2_2) / denominator2
        
        # Total physics loss
        physics_loss = tf.reduce_mean(residual1**2 + residual2**2)
        
        return physics_loss
    
    def initial_condition_loss(self, t0, initial_state):
        """Compute loss for initial conditions"""
        predicted_state = self.predict_state(t0)
        ic_loss = tf.reduce_mean((predicted_state - initial_state)**2)
        return ic_loss
    
    def total_loss(self, t, t0, initial_state, physics_weight=1.0, ic_weight=1.0):
        """Compute total loss combining physics and initial conditions"""
        physics_loss = self.physics_loss(t)
        ic_loss = self.initial_condition_loss(t0, initial_state)
        
        total_loss = physics_weight * physics_loss + ic_weight * ic_loss
        return total_loss, physics_loss, ic_loss
    
    def train(self, t_train, t0, initial_state, epochs=10000, learning_rate=0.001, 
              physics_weight=1.0, ic_weight=10.0):
        """Train the PINN"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        history = {'total_loss': [], 'physics_loss': [], 'ic_loss': []}
        
        print("Training PINN...")
        start_time = time.time()
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                total_loss, physics_loss, ic_loss = self.total_loss(
                    t_train, t0, initial_state, physics_weight, ic_weight
                )
            
            # Compute gradients
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # Store history
            history['total_loss'].append(float(total_loss))
            history['physics_loss'].append(float(physics_loss))
            history['ic_loss'].append(float(ic_loss))
            
            # Print progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss:.6f}, "
                      f"Physics Loss = {physics_loss:.6f}, IC Loss = {ic_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history

def generate_true_solution(t, initial_state, L1, L2, m1, m2):
    """Generate true solution using scipy for comparison"""
    def derivatives(y, t, L1, L2, m1, m2):
        theta1, z1, theta2, z2 = y
        c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
        theta1_dot = z1
        z1_dot = (m2*9.81*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - (m1+m2)*9.81*np.sin(theta1)) / L1 / (m1 + m2*s**2)
        theta2_dot = z2
        z2_dot = ((m1+m2)*(L1*z1**2*s - 9.81*np.sin(theta2) + 9.81*np.sin(theta1)*c) + m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
        return theta1_dot, z1_dot, theta2_dot, z2_dot
    
    states = odeint(derivatives, initial_state, t, args=(L1, L2, m1, m2))
    return states

def plot_results(t, true_states, predicted_states, history):
    """Plot the results"""
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Training loss
    plt.subplot(2, 3, 1)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['physics_loss'], label='Physics Loss')
    plt.plot(history['ic_loss'], label='IC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.yscale('log')
    
    # State variables comparison
    state_names = ['θ₁', 'θ₁̇', 'θ₂', 'θ₂̇']
    for i in range(4):
        plt.subplot(2, 3, i+2)
        plt.plot(t, true_states[:, i], 'b-', label='True', linewidth=2)
        plt.plot(t, predicted_states[:, i], 'r--', label='PINN', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel(state_names[i])
        plt.legend()
        plt.title(f'{state_names[i]} vs Time')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_animation(t, true_states, predicted_states, L1, L2):
    """Create animation comparing true and predicted pendulum motion"""
    # Extract angles
    theta1_true = true_states[:, 0]
    theta2_true = true_states[:, 2]
    theta1_pred = predicted_states[:, 0]
    theta2_pred = predicted_states[:, 2]
    
    # Compute coordinates
    x1_true = L1 * np.sin(theta1_true)
    y1_true = -L1 * np.cos(theta1_true)
    x2_true = x1_true + L2 * np.sin(theta2_true)
    y2_true = y1_true - L2 * np.cos(theta2_true)
    
    x1_pred = L1 * np.sin(theta1_pred)
    y1_pred = -L1 * np.cos(theta1_pred)
    x2_pred = x1_pred + L2 * np.sin(theta2_pred)
    y2_pred = y1_pred - L2 * np.cos(theta2_pred)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    line_true, = ax.plot([], [], 'o-', lw=3, color='blue', label='True', markersize=8)
    line_pred, = ax.plot([], [], 'o-', lw=3, color='red', label='PINN', markersize=8)
    
    def init():
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Double Pendulum: True vs PINN Prediction')
        return line_true, line_pred
    
    def animate(i):
        # Update true pendulum
        thisx_true = [0, x1_true[i], x2_true[i]]
        thisy_true = [0, y1_true[i], y2_true[i]]
        line_true.set_data(thisx_true, thisy_true)
        
        # Update predicted pendulum
        thisx_pred = [0, x1_pred[i], x2_pred[i]]
        thisy_pred = [0, y1_pred[i], y2_pred[i]]
        line_pred.set_data(thisx_pred, thisy_pred)
        
        return line_true, line_pred
    
    # Create animation with reduced frame rate for better visualization
    step = max(1, len(t) // 200)  # Show ~200 frames
    ani = animation.FuncAnimation(fig, animate, frames=range(0, len(t), step),
                                init_func=init, blit=True, interval=50)
    
    plt.show()
    return ani

def main():
    """Main function to run the PINN double pendulum simulation"""
    print("Physics-Informed Neural Network for Double Pendulum")
    print("=" * 50)
    
    # System parameters
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    
    # Time domain
    t_max = 10.0
    n_points = 1000
    t = np.linspace(0, t_max, n_points)
    
    # Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
    initial_state = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
    
    # Generate true solution for comparison
    print("Generating true solution...")
    true_states = generate_true_solution(t, initial_state, L1, L2, m1, m2)
    
    # Initialize and train PINN
    print("Initializing PINN...")
    pinn = DoublePendulumPINN(L1=L1, L2=L2, m1=m1, m2=m2, g=g)
    
    # Training parameters
    epochs = 5000
    learning_rate = 0.001
    physics_weight = 1.0
    ic_weight = 10.0
    
    # Train the model
    history = pinn.train(t, t[0], initial_state, epochs=epochs, 
                        learning_rate=learning_rate,
                        physics_weight=physics_weight, 
                        ic_weight=ic_weight)
    
    # Generate predictions
    print("Generating predictions...")
    predicted_states = pinn.predict_state(t).numpy()
    
    # Plot results
    print("Plotting results...")
    plot_results(t, true_states, predicted_states, history)
    
    # Create animation
    print("Creating animation...")
    ani = create_animation(t, true_states, predicted_states, L1, L2)
    
    # Calculate errors
    mse = np.mean((true_states - predicted_states)**2, axis=0)
    print("\nMean Squared Error for each state variable:")
    state_names = ['θ₁', 'θ₁̇', 'θ₂', 'θ₂̇']
    for i, name in enumerate(state_names):
        print(f"{name}: {mse[i]:.6f}")
    
    print(f"\nTotal MSE: {np.mean(mse):.6f}")
    
    return pinn, history, true_states, predicted_states

if __name__ == "__main__":
    pinn, history, true_states, predicted_states = main() 