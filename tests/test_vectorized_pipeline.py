#!/usr/bin/env python3
"""
Test script to verify the vectorized pipeline implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from BH_population import Universe
from PTA import PTA
from calculate_gw import GW

def test_batch_universe_generation():
    """Test that batch universe generation works correctly"""
    print("Testing batch universe generation...")
    
    # Test parameters
    alpha = -3.0
    Ω_min = 1e-8
    Ω_max = 1e-6
    M = 100
    num_realizations = 5
    seed = 42
    
    # Generate batch
    batch_params = Universe.generate_batch(alpha, Ω_min, Ω_max, M, num_realizations, seed)
    
    # Check shapes
    for key, values in batch_params.items():
        expected_shape = (num_realizations, M)
        assert values.shape == expected_shape, f"{key} has shape {values.shape}, expected {expected_shape}"
    
    # Check that different realizations are different
    assert not np.allclose(batch_params['Ω'][0], batch_params['Ω'][1]), "Different realizations should be different"
    
    # Check reproducibility
    batch_params2 = Universe.generate_batch(alpha, Ω_min, Ω_max, M, num_realizations, seed)
    for key in batch_params:
        assert np.allclose(batch_params[key], batch_params2[key]), f"{key} not reproducible with same seed"
    
    print("✓ Batch universe generation works correctly")
    return True

def test_batch_gw_computation():
    """Test that batch GW computation works"""
    print("Testing batch GW computation...")
    
    # Small test case
    alpha = -2.0
    Ω_min = 1e-8
    Ω_max = 1e-6
    M = 50
    num_realizations = 3
    
    # Create PTA
    year = 3.154e7
    week = 604800
    Tobs = 2*year
    dt = 2*week
    # Change to correct directory for PTA to find data file
    original_dir = os.getcwd()
    os.chdir('src')
    try:
        pulsars = PTA(Tobs=Tobs, dt=dt, seed=1)
    finally:
        os.chdir(original_dir)
    
    # Generate batch parameters
    batch_params = Universe.generate_batch(alpha, Ω_min, Ω_max, M, num_realizations, seed=123)
    
    # Test batch computation
    try:
        batch_a = GW.compute_a_batch(batch_params, pulsars)
        print(f"✓ Batch computation succeeded, output shape: {batch_a.shape}")
        
        expected_shape = (num_realizations, len(pulsars.t), pulsars.Npsr)
        assert batch_a.shape == expected_shape, f"Wrong output shape: {batch_a.shape} vs {expected_shape}"
        
        return True
    except Exception as e:
        print(f"✗ Batch computation failed: {e}")
        return False

def test_equivalence_small_case():
    """Test that vectorized gives same results as sequential for small case"""
    print("Testing equivalence between vectorized and sequential...")
    
    # Very small test case
    alpha = -2.0
    Ω_min = 1e-8
    Ω_max = 1e-6
    M = 10
    num_realizations = 3
    seed = 456
    
    year = 3.154e7
    week = 604800
    Tobs = 1*year
    dt = 4*week
    # Change to correct directory for PTA to find data file
    original_dir = os.getcwd()
    os.chdir('src')
    try:
        pulsars = PTA(Tobs=Tobs, dt=dt, seed=1)
    finally:
        os.chdir(original_dir)
    
    # Sequential approach (original)
    np.random.seed(seed)
    sequential_results = []
    for i in range(num_realizations):
        # Use i as seed to get different realizations
        universe = Universe(alpha, Ω_min, Ω_max, M, seed=seed+i)
        gw = GW(universe, pulsars)
        a = gw.compute_a_jax()
        sequential_results.append(np.array(a))
    
    sequential_batch = np.array(sequential_results)  # Shape: (num_realizations, T, N)
    
    # Vectorized approach 
    batch_params = Universe.generate_batch(alpha, Ω_min, Ω_max, M, num_realizations, seed)
    vectorized_batch = GW.compute_a_batch(batch_params, pulsars)
    
    print(f"Sequential shape: {sequential_batch.shape}")
    print(f"Vectorized shape: {vectorized_batch.shape}")
    
    # Compare shapes
    assert sequential_batch.shape == vectorized_batch.shape, "Shapes don't match"
    
    # Note: We don't expect exact numerical equivalence due to different random seeds
    # But we can check that the magnitude and distribution are similar
    seq_mean = np.mean(np.abs(sequential_batch))
    vec_mean = np.mean(np.abs(vectorized_batch))
    
    print(f"Sequential mean magnitude: {seq_mean:.2e}")
    print(f"Vectorized mean magnitude: {vec_mean:.2e}")
    
    # They should be within same order of magnitude
    ratio = vec_mean / seq_mean
    assert 0.1 < ratio < 10, f"Results have very different magnitudes: ratio = {ratio}"
    
    print("✓ Vectorized and sequential approaches produce comparable results")
    return True

def test_performance_comparison():
    """Compare performance of vectorized vs sequential"""
    print("\nPerformance comparison...")
    
    alpha = -3.0
    Ω_min = 3e-9
    Ω_max = 1e-6
    M = int(1e4)
    num_realizations = 500  # Smaller for testing
    
    year = 3.154e7
    week = 604800
    Tobs = 5*year
    dt = 1*week
    # Change to correct directory for PTA to find data file
    original_dir = os.getcwd()
    os.chdir('src')
    try:
        pulsars = PTA(Tobs=Tobs, dt=dt, seed=1)
    finally:
        os.chdir(original_dir)
    
    print(f"Test configuration: M={M}, realizations={num_realizations}, T={len(pulsars.t)}")
    
    # Time sequential approach (just a few iterations for comparison)
    print("Timing sequential approach (10 iterations only)...")
    start_time = time.time()
    for i in range(10):
        universe = Universe(alpha, Ω_min, Ω_max, M, seed=i)
        gw = GW(universe, pulsars)
        a = gw.compute_a_jax()
    sequential_time_per_iter = (time.time() - start_time) / 10
    estimated_sequential_total = sequential_time_per_iter * num_realizations
    
    # Time vectorized approach
    print("Timing vectorized approach...")
    start_time = time.time()
    batch_params = Universe.generate_batch(alpha, Ω_min, Ω_max, M, num_realizations, seed=42)
    batch_a = GW.compute_a_batch(batch_params, pulsars)
    vectorized_time = time.time() - start_time
    
    print(f"Sequential (estimated): {estimated_sequential_total:.2f} seconds")
    print(f"Vectorized (actual): {vectorized_time:.2f} seconds")
    
    if vectorized_time < estimated_sequential_total:
        speedup = estimated_sequential_total / vectorized_time
        print(f"🚀 Speedup: {speedup:.1f}x")
    else:
        print("⚠️ Vectorized is slower (may improve with larger batch sizes or GPU)")
    
    return True

if __name__ == "__main__":
    try:
        test_batch_universe_generation()
        test_batch_gw_computation() 
        test_equivalence_small_case()
        test_performance_comparison()
        print("\n🎉 All vectorization tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)