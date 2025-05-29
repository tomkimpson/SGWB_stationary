#!/usr/bin/env python3
"""
Test script to verify the updated plotting functions work with show_fig and save_fig parameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import the updated plotting functions
from main import plot_1d, plot_2d

def test_plotting_functions():
    """Test that plotting functions work with new parameters"""
    
    print("Testing updated plotting functions...")
    
    # Create some test data
    T = 100
    t = np.linspace(0, 365*24*3600*10, T)  # 10 years
    
    # Test data for plot_1d
    num_realisations = 5
    array_1D = np.random.randn(T, num_realisations) * 0.1
    
    # Test data for plot_2d  
    array_2D = np.random.randn(T, T) * 0.1
    
    print("Testing plot_1d with save_fig=True, show_fig=False...")
    try:
        plot_1d(t, array_1D, show_fig=False, save_fig=True)
        if os.path.exists('outputs/plot_1d.png'):
            print("✓ plot_1d saved successfully to outputs/plot_1d.png")
        else:
            print("✗ plot_1d failed to save")
            return False
    except Exception as e:
        print(f"✗ plot_1d failed with error: {e}")
        return False
    
    print("Testing plot_2d with save_fig=True, show_fig=False...")
    try:
        plot_2d(t, array_2D, show_fig=False, save_fig=True)
        if os.path.exists('outputs/plot_2d.png'):
            print("✓ plot_2d saved successfully to outputs/plot_2d.png")
        else:
            print("✗ plot_2d failed to save")
            return False
    except Exception as e:
        print(f"✗ plot_2d failed with error: {e}")
        return False
    
    print("Testing plot_1d with save_fig=False, show_fig=False...")
    try:
        plot_1d(t, array_1D, show_fig=False, save_fig=False)
        print("✓ plot_1d ran without display or save")
    except Exception as e:
        print(f"✗ plot_1d failed with error: {e}")
        return False
    
    print("Testing plot_2d with save_fig=False, show_fig=False...")
    try:
        plot_2d(t, array_2D, show_fig=False, save_fig=False)
        print("✓ plot_2d ran without display or save")
    except Exception as e:
        print(f"✗ plot_2d failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_plotting_functions()
    if success:
        print("\n🎉 All plotting tests passed!")
    else:
        print("\n❌ Plotting tests failed!")
        sys.exit(1)