# -*- coding: utf-8 -*-
"""
Test script for PINN Double Pendulum implementation
This script tests the basic functionality without running the full training
"""

import numpy as np
import matplotlib.pyplot as plt

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  Version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        from scipy.integrate import odeint
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_pinn_class():
    """Test the PINN class initialization and basic methods"""
    try:
        from PINN_Double_Pendulum_Simple import DoublePendulumPINN
        
        # Test initialization
        pinn = DoublePendulumPINN(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81)
        print("✓ PINN class initialized successfully")
        
        # Test prediction
        t_test = np.array([0.0, 1.0, 2.0])
        predictions = pinn.predict_state(t_test)
        print(f"✓ Predictions shape: {predictions.shape}")
        
        # Test physics loss computation
        physics_loss = pinn.physics_loss(t_test)
        print(f"✓ Physics loss computed: {float(physics_loss):.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ PINN class test failed: {e}")
        return False

def test_true_solution():
    """Test the true solution generation"""
    try:
        from PINN_Double_Pendulum_Simple import generate_true_solution
        
        # Test parameters
        t = np.linspace(0, 5, 100)
        initial_state = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
        L1, L2, m1, m2 = 1.0, 1.0, 1.0, 1.0
        
        # Generate solution
        states = generate_true_solution(t, initial_state, L1, L2, m1, m2)
        print(f"✓ True solution generated, shape: {states.shape}")
        
        # Check that solution is reasonable
        assert states.shape == (len(t), 4), "Wrong output shape"
        assert not np.any(np.isnan(states)), "Solution contains NaN values"
        assert not np.any(np.isinf(states)), "Solution contains infinite values"
        
        return True
        
    except Exception as e:
        print(f"✗ True solution test failed: {e}")
        return False

def test_visualization():
    """Test basic plotting functionality"""
    try:
        # Generate test data
        t = np.linspace(0, 5, 100)
        true_states = np.random.randn(len(t), 4) * 0.1
        predicted_states = true_states + np.random.randn(len(t), 4) * 0.05
        
        history = {
            'total_loss': np.exp(-np.linspace(0, 5, 100)),
            'physics_loss': np.exp(-np.linspace(0, 5, 100)) * 0.5,
            'ic_loss': np.exp(-np.linspace(0, 5, 100)) * 0.3
        }
        
        # Test plotting (without showing)
        from PINN_Double_Pendulum_Simple import plot_results
        plot_results(t, true_states, predicted_states, history)
        plt.close('all')  # Close all figures
        print("✓ Plotting functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def run_quick_training():
    """Run a very quick training to test the full pipeline"""
    try:
        from PINN_Double_Pendulum_Simple import DoublePendulumPINN, generate_true_solution
        
        print("\nRunning quick training test...")
        
        # Setup
        t = np.linspace(0, 2, 50)  # Shorter time, fewer points
        initial_state = np.array([np.pi/2, 0.0, np.pi/2, 0.0])
        
        # Generate true solution
        true_states = generate_true_solution(t, initial_state, 1.0, 1.0, 1.0, 1.0)
        
        # Initialize PINN
        pinn = DoublePendulumPINN()
        
        # Quick training (only 10 epochs)
        history = pinn.train(t, t[0], initial_state, epochs=10, learning_rate=0.01)
        
        # Test prediction
        predicted_states = pinn.predict_state(t).numpy()
        
        # Calculate basic error
        mse = np.mean((true_states - predicted_states)**2)
        print(f"✓ Quick training completed, MSE: {mse:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing PINN Double Pendulum Implementation")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("PINN Class", test_pinn_class),
        ("True Solution Generation", test_true_solution),
        ("Visualization Functions", test_visualization),
        ("Quick Training", run_quick_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The PINN implementation is ready to use.")
        print("\nTo run the full simulation:")
        print("python PINN_Double_Pendulum_Simple.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 