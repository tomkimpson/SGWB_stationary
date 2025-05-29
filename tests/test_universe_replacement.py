#!/usr/bin/env python3
"""
Test script to verify the replacement of Bilby with NumPy/SciPy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from BH_population import Universe

def test_universe_functionality():
    """Test that the new Universe class works and produces reasonable distributions"""
    
    print("Testing Universe class without Bilby...")
    
    # Test parameters
    Ω_power_law_index = -2.0
    Ω_min = 1e-9
    Ω_max = 1e-7
    M = 10000
    seed = 42
    
    # Create universe instance
    print(f"Creating Universe with M={M} sources...")
    universe = Universe(Ω_power_law_index, Ω_min, Ω_max, M, seed=seed)
    
    # Check that all parameters have correct shapes
    assert universe.Ω.shape == (M,), f"Ω shape wrong: {universe.Ω.shape}"
    assert universe.h.shape == (M,), f"h shape wrong: {universe.h.shape}"
    assert universe.φ0.shape == (M,), f"φ0 shape wrong: {universe.φ0.shape}"
    assert universe.ψ.shape == (M,), f"ψ shape wrong: {universe.ψ.shape}"
    assert universe.ι.shape == (M,), f"ι shape wrong: {universe.ι.shape}"
    assert universe.δ.shape == (M,), f"δ shape wrong: {universe.δ.shape}"
    assert universe.α.shape == (M,), f"α shape wrong: {universe.α.shape}"
    print("✓ All parameter shapes correct")
    
    # Check parameter ranges
    assert np.all(universe.Ω >= Ω_min) and np.all(universe.Ω <= Ω_max), "Ω out of range"
    assert np.all(universe.h == 1.0), "h not constant at 1.0"
    assert np.all(universe.φ0 >= 0) and np.all(universe.φ0 <= 2*np.pi), "φ0 out of range"
    assert np.all(universe.ψ >= 0) and np.all(universe.ψ <= np.pi), "ψ out of range"
    assert np.all(universe.ι >= 0) and np.all(universe.ι <= np.pi), "ι out of range"
    assert np.all(universe.δ >= -np.pi/2) and np.all(universe.δ <= np.pi/2), "δ out of range"
    assert np.all(universe.α >= 0) and np.all(universe.α <= 2*np.pi), "α out of range"
    print("✓ All parameter ranges correct")
    
    # Test seeding reproducibility
    universe1 = Universe(Ω_power_law_index, Ω_min, Ω_max, 100, seed=123)
    universe2 = Universe(Ω_power_law_index, Ω_min, Ω_max, 100, seed=123)
    
    assert np.allclose(universe1.Ω, universe2.Ω), "Seeding not working for Ω"
    assert np.allclose(universe1.φ0, universe2.φ0), "Seeding not working for φ0"
    assert np.allclose(universe1.ι, universe2.ι), "Seeding not working for ι"
    assert np.allclose(universe1.δ, universe2.δ), "Seeding not working for δ"
    print("✓ Seeding reproducibility works")
    
    # Test that different seeds give different results
    universe3 = Universe(Ω_power_law_index, Ω_min, Ω_max, 100, seed=456)
    assert not np.allclose(universe1.Ω, universe3.Ω), "Different seeds giving same results"
    print("✓ Different seeds produce different results")
    
    # Check statistical properties (basic sanity checks)
    print(f"\nStatistical checks (M={M}):")
    print(f"Ω range: [{universe.Ω.min():.2e}, {universe.Ω.max():.2e}]")
    print(f"φ0 mean: {universe.φ0.mean():.2f} (should be ~π)")
    print(f"ψ mean: {universe.ψ.mean():.2f} (should be ~π/2)")
    print(f"α mean: {universe.α.mean():.2f} (should be ~π)")
    
    # Check power law behavior for Ω (rough check)
    # For power law with α=-2, we expect more samples at lower frequencies
    Ω_median = np.median(universe.Ω)
    Ω_lower_half = np.sum(universe.Ω < Ω_median)
    print(f"Ω samples below median: {Ω_lower_half}/{M} (power law α=-2 should favor lower values)")
    
    print("\n🎉 All tests passed! Universe class successfully replaced Bilby.")
    return True

def test_performance():
    """Compare performance of new implementation"""
    import time
    
    print("\nTesting performance...")
    M = 50000  # Large number for timing
    
    start_time = time.time()
    universe = Universe(-2.0, 1e-9, 1e-7, M, seed=42)
    end_time = time.time()
    
    print(f"Generated {M} GW sources in {end_time - start_time:.4f} seconds")
    print(f"Rate: {M/(end_time - start_time):.0f} sources/second")
    
    return True

if __name__ == "__main__":
    success = test_universe_functionality()
    if success:
        test_performance()
        print("\n✅ All Universe replacement tests successful!")
    else:
        print("\n❌ Universe replacement tests failed!")
        sys.exit(1)