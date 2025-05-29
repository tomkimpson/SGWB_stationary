#!/usr/bin/env python3
"""
Test script to verify JAX optimization of compute_a() function
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from calculate_gw import GW

# Simple mock classes for testing
class MockUniverse:
    def __init__(self, M=100):
        self.M = M
        self.Ω = np.random.uniform(1e-8, 1e-7, M)
        self.δ = np.random.uniform(-np.pi/2, np.pi/2, M)
        self.α = np.random.uniform(0, 2*np.pi, M)
        self.ψ = np.random.uniform(0, np.pi, M)
        self.h = np.random.uniform(1e-15, 1e-14, M)
        self.ι = np.random.uniform(0, np.pi, M)
        self.φ0 = np.random.uniform(0, 2*np.pi, M)

class MockPTA:
    def __init__(self, Npsr=20, Nt=100):
        self.Npsr = Npsr
        self.t = np.linspace(0, 365*24*3600, Nt)  # 1 year in seconds
        self.q = np.random.randn(3, Npsr)
        self.q = self.q / np.linalg.norm(self.q, axis=0)  # Normalize
        self.q_products = np.array([self.q[i] * self.q[j] for i in range(3) for j in range(3)]).reshape(9, Npsr)
        self.d_psr = np.random.uniform(0.1, 10, Npsr) * 3.086e19  # distances in meters
        self.d_psr = self.d_psr.reshape(Npsr, 1)  # Shape (Npsr, 1) like in real PTA

def test_jax_optimization():
    print("Testing JAX optimization of compute_a()...")
    
    # Create test data with larger problem size to see JAX benefits
    universe = MockUniverse(M=5000)  # Larger M for better performance testing
    pta = MockPTA(Npsr=20, Nt=500)
    
    # Create GW instance
    gw = GW(universe, pta)
    
    print(f"Test configuration: M={universe.M}, Npsr={pta.Npsr}, Nt={len(pta.t)}")
    
    # Time original implementation
    print("\nTiming original numpy implementation...")
    start_time = time.time()
    result_numpy = gw.compute_a()
    numpy_time = time.time() - start_time
    print(f"NumPy implementation: {numpy_time:.4f} seconds")
    
    # Time JAX implementation
    print("\nTiming JAX implementation...")
    start_time = time.time()
    result_jax = gw.compute_a_jax()
    jax_time = time.time() - start_time
    print(f"JAX implementation (first run): {jax_time:.4f} seconds")
    
    # Time JAX implementation again (should be faster due to JIT compilation)
    start_time = time.time()
    result_jax2 = gw.compute_a_jax()
    jax_time2 = time.time() - start_time
    print(f"JAX implementation (second run): {jax_time2:.4f} seconds")
    
    # Check results are close
    max_diff = np.max(np.abs(result_numpy - np.array(result_jax)))
    print(f"\nMaximum difference between implementations: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ Results match within numerical precision")
    else:
        print("✗ Results differ significantly")
        return False
    
    # Performance comparison
    speedup = numpy_time / jax_time2
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 1:
        print("✓ JAX implementation is faster")
    else:
        print("⚠ JAX implementation is slower (may improve with larger problem sizes)")
    
    return True

if __name__ == "__main__":
    success = test_jax_optimization()
    if success:
        print("\n🎉 JAX optimization successful!")
    else:
        print("\n❌ JAX optimization failed!")
        sys.exit(1)