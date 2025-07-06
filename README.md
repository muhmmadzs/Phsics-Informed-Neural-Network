# Physics-Informed Neural Network (PINN) for Double Pendulum

This repository contains implementations of Physics-Informed Neural Networks (PINNs) for modeling the double pendulum system. The PINN approach combines the power of neural networks with the constraints of physical laws to create more accurate and physically consistent models.

## Overview

The double pendulum is a classic chaotic system that consists of two pendulums connected in series. The system is described by a set of coupled nonlinear differential equations that make it an excellent test case for physics-informed machine learning approaches.

## Files Description

### 1. `PINN_Double_Pendulum_Simple.py`
A simplified and well-documented implementation of a PINN for the double pendulum system. This version includes:

- **Neural Network Architecture**: A feedforward neural network that takes time as input and outputs the state variables
- **Physics-Informed Loss**: Incorporates the equations of motion directly into the loss function
- **Automatic Differentiation**: Uses TensorFlow's automatic differentiation to compute time derivatives
- **Training Loop**: Custom training loop with separate physics and initial condition losses
- **Visualization**: Comprehensive plotting and animation capabilities

### 2. `LSTM_Double_Pendulum_Phys.py`
An LSTM-based approach with physics-informed layers, showing an alternative way to incorporate physics constraints.

### 3. `LSTM_Double_Pendulum.py`
A baseline LSTM implementation without physics constraints for comparison.

### 4. `requirements.txt`
Lists all required dependencies for running the PINN implementations.

## Physics Background

### Double Pendulum Equations of Motion

The double pendulum system is described by the following coupled differential equations:

For the first pendulum:
```
θ₁̈ = [m₂g sin(θ₂)cos(θ₁-θ₂) - m₂sin(θ₁-θ₂)(L₁θ₁̇²cos(θ₁-θ₂) + L₂θ₂̇²) - (m₁+m₂)g sin(θ₁)] / [L₁(m₁ + m₂sin²(θ₁-θ₂))]
```

For the second pendulum:
```
θ₂̈ = [(m₁+m₂)(L₁θ₁̇²sin(θ₁-θ₂) - g sin(θ₂) + g sin(θ₁)cos(θ₁-θ₂)) + m₂L₂θ₂̇²sin(θ₁-θ₂)cos(θ₁-θ₂)] / [L₂(m₁ + m₂sin²(θ₁-θ₂))]
```

Where:
- θ₁, θ₂: Angular positions of the pendulums
- θ₁̇, θ₂̇: Angular velocities
- θ₁̈, θ₂̈: Angular accelerations
- L₁, L₂: Lengths of pendulum arms
- m₁, m₂: Masses of pendulum bobs
- g: Gravitational acceleration

## PINN Implementation Details

### Neural Network Architecture
```python
model = keras.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(1,)),  # Time input
    layers.Dense(64, activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(4)  # Output: [θ₁, θ₁̇, θ₂, θ₂̇]
])
```

### Physics-Informed Loss Function
The total loss combines two components:

1. **Physics Loss**: Ensures the neural network predictions satisfy the equations of motion
2. **Initial Condition Loss**: Ensures the network satisfies the given initial conditions

```python
total_loss = physics_weight * physics_loss + ic_weight * initial_condition_loss
```

### Automatic Differentiation
The implementation uses TensorFlow's automatic differentiation to compute:
- First derivatives: θ₁̇, θ₂̇
- Second derivatives: θ₁̈, θ₂̈

This allows the physics constraints to be enforced without needing to derive analytical expressions for the derivatives.

## Installation and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the PINN Implementation
```bash
python PINN_Double_Pendulum_Simple.py
```

### 3. Expected Output
The script will:
- Train the PINN for 5000 epochs
- Display training progress and loss values
- Generate comparison plots between true and predicted solutions
- Create an animation showing the pendulum motion
- Print mean squared error metrics

## Key Features

### 1. Physics-Informed Training
- Incorporates physical laws directly into the loss function
- Uses automatic differentiation for derivative computation
- Balances physics constraints with initial conditions

### 2. Comprehensive Visualization
- Training loss history plots
- State variable comparisons
- Animated pendulum motion
- Error analysis

### 3. Flexible Architecture
- Easily modifiable system parameters (masses, lengths, gravity)
- Adjustable training parameters (learning rate, epochs, loss weights)
- Extensible to other dynamical systems

## Advantages of PINN Approach

1. **Physical Consistency**: The model is constrained by known physical laws
2. **Data Efficiency**: Requires less training data than pure data-driven approaches
3. **Generalization**: Better extrapolation beyond training data
4. **Interpretability**: The model learns physically meaningful representations
5. **Robustness**: More stable predictions due to physics constraints

## Comparison with Traditional Methods

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| **PINN** | Physics-consistent, data-efficient, generalizable | Computationally intensive, requires physics knowledge |
| **LSTM** | Good at sequence modeling, flexible | No physics constraints, requires more data |
| **Numerical Integration** | Exact solution, fast | Not learnable, requires complete system knowledge |

## Customization

### Modifying System Parameters
```python
# Change pendulum properties
pinn = DoublePendulumPINN(L1=1.5, L2=1.0, m1=2.0, m2=1.0, g=9.81)
```

### Adjusting Training Parameters
```python
# Modify training settings
history = pinn.train(t, t[0], initial_state, 
                    epochs=10000,           # More training epochs
                    learning_rate=0.0005,   # Lower learning rate
                    physics_weight=2.0,     # Emphasize physics
                    ic_weight=5.0)          # Reduce IC emphasis
```

### Changing Initial Conditions
```python
# Different initial conditions
initial_state = np.array([np.pi/4, 1.0, np.pi/3, -0.5])
```

## Troubleshooting

### Common Issues

1. **Training Divergence**: Reduce learning rate or increase physics weight
2. **Poor Convergence**: Increase number of epochs or adjust loss weights
3. **Memory Issues**: Reduce batch size or number of time points
4. **Import Errors**: Ensure all dependencies are installed correctly

### Performance Tips

1. **GPU Acceleration**: Use GPU-enabled TensorFlow for faster training
2. **Batch Processing**: Process time points in batches for large datasets
3. **Early Stopping**: Implement early stopping based on validation loss
4. **Learning Rate Scheduling**: Use adaptive learning rates

## Future Extensions

1. **Multi-scale PINNs**: Handle systems with multiple time scales
2. **Uncertainty Quantification**: Add uncertainty estimates to predictions
3. **Adaptive Sampling**: Dynamically adjust training points
4. **Hybrid Approaches**: Combine PINNs with traditional numerical methods
5. **Control Applications**: Extend to optimal control problems

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.

2. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. SIAM Review, 63(1), 208-228.

3. Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. Advances in neural information processing systems, 31.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or feature requests. 
