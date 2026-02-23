"""
Test script for pairplot_posterior functionality.
Run this to verify that the pairplot generation works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.plot_manager import PlotManager

def test_pairplot():
    """Test pairplot generation with dummy data."""
    print("=" * 60)
    print("Testing pairplot_posterior functionality")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create plot manager
    plot_mgr = PlotManager(str(output_dir))
    
    # Generate dummy posterior samples (M1: 5 parameters)
    np.random.seed(42)
    n_samples = 1000
    n_params = 5
    
    # True values
    theta_true = np.array([0.8, 2.0, 1.0, 0.1, 0.2])
    
    # Generate samples around true values
    samples = np.random.randn(n_samples, n_params)
    samples = samples * 0.3 + theta_true[None, :]
    
    # MAP and mean estimates
    theta_MAP = np.array([0.75, 2.1, 0.95, 0.12, 0.18])
    theta_mean = np.array([0.78, 2.05, 0.98, 0.11, 0.19])
    
    # Parameter names
    param_names = ["a11", "a12", "a22", "b1", "b2"]
    
    print(f"\nGenerated test data:")
    print(f"  Samples shape: {samples.shape}")
    print(f"  True values: {theta_true}")
    print(f"  MAP values: {theta_MAP}")
    print(f"  Mean values: {theta_mean}")
    print(f"  Parameter names: {param_names}")
    
    try:
        print("\nGenerating pairplot...")
        plot_mgr.plot_pairplot_posterior(
            samples, theta_true, theta_MAP, theta_mean,
            param_names, "M1_test"
        )
        print(f"[SUCCESS] pairplot_posterior_M1_test.png saved to {output_dir}/")
        
        # Check if file was created
        output_file = output_dir / "pairplot_posterior_M1_test.png"
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"   File size: {file_size:,} bytes")
        else:
            print("   [WARNING] Output file not found!")
            
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_pairplot()
    sys.exit(0 if success else 1)
