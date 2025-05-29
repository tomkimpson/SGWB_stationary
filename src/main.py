#!/usr/bin/env python3
"""
Stochastic Gravitational Wave Background (SGWB) Analysis Pipeline

This script computes the stochastic gravitational wave background from a population 
of binary black holes and analyzes the cross-correlation signals in a pulsar timing array.

Main components:
- Universe: Population of M black hole binaries with random GW parameters
- PTA: Pulsar timing array with N pulsars  
- GW: Gravitational wave signal computation (JAX-optimized)
- Analysis: Cross-correlation products and statistical analysis
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
from BH_population import Universe
from PTA import PTA 
from calculate_gw import GW
from plotting import plot_1d, plot_2d

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

def main():
    """Main SGWB analysis pipeline"""
    
    print("="*60)
    print("STOCHASTIC GRAVITATIONAL WAVE BACKGROUND ANALYSIS")
    print("="*60)
    
    # ==========================================
    # JAX CONFIGURATION
    # ==========================================
    
    print("JAX Configuration:")
    print(f"  64-bit precision: {jax.config.jax_enable_x64}")
    
    # Check for GPU availability
    devices = jax.devices()
    print(f"  Available devices: {[str(d) for d in devices]}")
    
    if any("gpu" in str(device).lower() for device in devices):
        print("  ✓ GPU detected - JAX will use GPU acceleration")
    else:
        print("  ⚠ No GPU detected - using CPU only")
    
    print(f"  Default backend: {jax.default_backend()}")
    print()
    
    # ==========================================
    # SIMULATION PARAMETERS
    # ==========================================
    
    # Physical constants
    year = 3.154e7  # seconds
    week = 604800   # seconds
    
    # Black hole population parameters
    alpha = -3.0                    # Power law exponent for frequency distribution
    Tobs = 10 * year               # Observation period (10 years)
    dt = 1 * week                  # Cadence (1 week)
    Ω_min = 1 / (10 * year)        # Minimum GW frequency (set by Tobs)
    Ω_max = 1 / week               # Maximum GW frequency (set by dt)
    M = int(1e4)                   # Number of GW sources
    
    # Analysis parameters
    num_realisations = int(1e2)    # Number of universe realizations
    pulsar_seed = 1                # Seed for pulsar selection
    
    print(f"Simulation Parameters:")
    print(f"  Observation time: {Tobs/year:.1f} years")
    print(f"  Observation cadence: {dt/week:.1f} weeks")
    print(f"  GW frequency range: [{Ω_min:.2e}, {Ω_max:.2e}] Hz")
    print(f"  Number of GW sources: {M:,}")
    print(f"  Power law exponent: α = {alpha}")
    print(f"  Universe realizations: {num_realisations:,}")
    print()
    
    # ==========================================
    # PULSAR TIMING ARRAY SETUP
    # ==========================================
    
    print("Setting up Pulsar Timing Array...")
    pulsars = PTA(Tobs=Tobs, dt=dt, seed=pulsar_seed)
    num_times = len(pulsars.t)
    
    print(f"  Number of pulsars: {pulsars.Npsr}")
    print(f"  Number of time samples: {num_times}")
    print(f"  Time span: {pulsars.t[-1]/year:.1f} years")
    print()
    
    # ==========================================
    # STOCHASTIC BACKGROUND COMPUTATION
    # ==========================================
    
    print("Computing stochastic gravitational wave background...")
    print("(Using JAX-optimized GW calculations)")
    
    # Initialize arrays for results
    array_1D = np.zeros((num_times, num_realisations))  # Cross-correlation timeseries
    array_2D = np.zeros((num_times, num_times))         # 2D correlation matrix
    
    # Main computation loop
    print(f"Processing {num_realisations} universe realizations:")
    
    for i in tqdm(range(num_realisations), desc="Realizations"):
        # Generate universe realization with reproducible seed
        universe_i = Universe(alpha, Ω_min, Ω_max, M, seed=i)
        
        # Compute gravitational wave signals at each pulsar
        SGWB = GW(universe_i, pulsars)
        a = SGWB.compute_a_jax()  # JAX-optimized computation
        
        # Extract signals for the two pulsars
        a1 = a[:, 0]  # Pulsar 1 signal
        a2 = a[:, 1]  # Pulsar 2 signal
        
        # Compute cross-correlation products
        ac = a1[0]                              # Reference amplitude
        product = ac * a2                       # 1D cross-correlation product
        outer_product = np.outer(a1, a2)       # 2D correlation matrix
        
        # Store results
        array_1D[:, i] = product
        array_2D += outer_product
    
    print("✓ Computation completed successfully")
    print()
    
    # ==========================================
    # STATISTICAL ANALYSIS
    # ==========================================
    
    print("Statistical Analysis:")
    
    # 1D statistics
    mean_1D = np.mean(array_1D, axis=1)
    std_1D = np.std(array_1D, axis=1)
    
    print(f"  1D Cross-correlation:")
    print(f"    Mean amplitude: {np.mean(np.abs(mean_1D)):.2e}")
    print(f"    RMS variability: {np.mean(std_1D):.2e}")
    
    # 2D statistics
    array_2D_normalized = array_2D / num_realisations
    
    print(f"  2D Correlation matrix:")
    print(f"    Matrix size: {array_2D.shape[0]} × {array_2D.shape[1]}")
    print(f"    Mean value: {np.mean(array_2D_normalized):.2e}")
    print(f"    Max value: {np.max(array_2D_normalized):.2e}")
    print()
    
    # ==========================================
    # RESULTS AND PLOTTING
    # ==========================================
    
    print("Generating plots...")
    print("  Saving plot_1d.png (cross-correlation analysis)")
    print("  Saving plot_2d.png (correlation matrix)")
    
    # Generate plots (save to outputs/ directory)
    plot_1d(pulsars.t, array_1D, show_fig=False, save_fig=True)
    plot_2d(pulsars.t, array_2D, show_fig=False, save_fig=True)
    
    print("✓ Plots saved to outputs/ directory")
    print()
    
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved:")
    print(f"  • outputs/plot_1d.png - Cross-correlation timeseries analysis")
    print(f"  • outputs/plot_2d.png - 2D correlation matrix")
    print()

if __name__ == "__main__":
    main()