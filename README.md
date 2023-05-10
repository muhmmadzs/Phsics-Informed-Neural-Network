# Harmonic Oscillator Model and Animation

This repository contains code for building a LSTM model of a simple harmonic oscillator, and for creating an animation of the oscillator's motion.
# Description
The Python script harmonic_oscillator.py builds and trains a LSTM model on data generated from a simple harmonic oscillator. The oscillator is described by the following equations of motion:

dx/dt = v

dv/dt = -x*omega^2

where x is the position of the oscillator and v is its velocity.

The LSTM model is trained to predict the position and velocity of the oscillator at each time step, given its initial position and velocity.

Once the model is trained, the script generates an animation that shows the true and predicted motion of the oscillator. The animation includes a trace of the oscillator's motion over time, which is shown as a faint line. The oscillator's current position is shown as a dot.

# Requirements
Python 3.x

TensorFlow 2.x

NumPy

Matplotlib

# Usage

You can run the script using the following command:

'python file_name.py'

The script will train the LSTM model and then generate the animation. The animation will be displayed in a new window.
# More Example

Comming Soon .........
