# Harmonic Oscillator Model and Animation

This repository contains an example of using Physics Informed Machine Learning, specifically using a Long Short-Term Memory (LSTM) network, to predict the future states of a Dynamic system given its initial state. It also includes the code for animation of the results

# Description
The Python scripts  builds and trains a LSTM model on data generated from a simple harmonic oscillator and Double Pendulum. The both are described by the following equations of motion:

# Harmonic Oscillator

dx/dt = v

dv/dt = -x*omega^2

where x is the position of the oscillator and v is its velocity.

The LSTM model is trained to predict the position and velocity of the oscillator at each time step, given its initial position and velocity.

Once the model is trained, the script generates an animation that shows the true and predicted motion of the oscillator. The animation includes a trace of the oscillator's motion over time, which is shown as a faint line. The oscillator's current position is shown as a dot.

# Double Pendulum

The double pendulum system is modeled by a set of differential equations which are derived from the physical laws governing the system. The governing equations for the double pendulum are as follows:

θ₁_dot = z₁

z₁_dot = (m₂*g*sin(θ₂)cos(θ₁-θ₂) - m₂*sin(θ₁-θ₂)(L₁*z₁²*cos(θ₁-θ₂) + L₂*z₂²) - (m₁+m₂)*g*sin(θ₁)) / L₁ / (m₁ + m₂*sin²(θ₁-θ₂))

θ₂_dot = z₂

z₂_dot = ((m₁+m₂)(L₁*z₁²*sin(θ₁-θ₂) - g*sin(θ₂) + g*sin(θ₁)*cos(θ₁-θ₂)) + m₂*L₂*z₂²*sin(θ₁-θ₂)*cos(θ₁-θ₂)) / L₂ / (m₁ + m₂*sin²(θ₁-θ₂))

where θ₁ and θ₂ are the angles of the two pendulum arms, z₁ and z₂ are the corresponding angular velocities, L₁ and L₂ are the lengths of the pendulum arms, and m₁ and m₂ are the masses of the pendulum bobs.

An LSTM network is trained on the solution of these equations for a given initial state. The trained LSTM network can then be used to predict

# Requirements
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

# Usage

You can run the script using the following command:

'python file_name.py'

The script will train the LSTM model and then generate the animation. The animation will be displayed in a new window.

# Note 

Further improvements could be made by fine-tuning the LSTM network architecture and training process.
Additionally, other types of physical systems could be modeled and predicted using a similar approach.

# More Example

Comming Soon .........
