"""
Test pairplot functionality with real MCMC results.
Loads actual data from a run directory and generates pairplot.
"""

import numpy as np
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from visualization.plot_manager import PlotManager

def test_pairplot_with_real_data(run_dir: str = None):
    """Test pairplot generation with real MCMC results."""
    print("=" * 60)
    print("Testing pairplot_posterior with REAL MCMC data")
    print("=" * 60)
    
    # Use the actual run directory if provided
    if run_dir is None:
        run_dir = Path("_runs/m1_1000_20260118_083726_good")
    else:
        run_dir = Path(run_dir)
    
    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        print("Please provide a valid run directory path.")
        return False
    
    print(f"\nLoading data from: {run_dir}")
    
    # Load results
    results_file = run_dir / "results_MAP_linearization.npz"
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        return False
    
    try:
        data = np.load(results_file)
        
        # Extract M1 data
        samples_M1 = data['samples_M1']
        theta_true_full = data['theta_true']
        MAP_M1 = data['MAP_M1']
        mean_M1 = data['mean_M1']
        
        # Extract first 5 parameters for M1
        theta_true_M1 = theta_true_full[:5]
        
        print(f"\nLoaded data:")
        print(f"  Samples shape: {samples_M1.shape}")
        print(f"  True values (M1): {theta_true_M1}")
        print(f"  MAP values (M1): {MAP_M1}")
        print(f"  Mean values (M1): {mean_M1}")
        
        # Parameter names for M1
        param_names = ["a11", "a12", "a22", "b1", "b2"]
        
        # Create output directory (use the run's figures directory)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create plot manager
        plot_mgr = PlotManager(str(figures_dir))
        
        print(f"\nGenerating pairplot...")
        plot_mgr.plot_pairplot_posterior(
            samples_M1, theta_true_M1, MAP_M1, mean_M1,
            param_names, "M1"
        )
        
        # Check if file was created
        output_file = figures_dir / "pairplot_posterior_M1.png"
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"[SUCCESS] pairplot_posterior_M1.png saved to {figures_dir}/")
            print(f"   File size: {file_size:,} bytes")
            print(f"   Full path: {output_file.resolve()}")
            return True
        else:
            print(f"[WARNING] Output file not found: {output_file}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Allow command line argument for run directory
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("\nUsage: python test_pairplot_with_real_data.py [run_directory]")
    print("Example: python test_pairplot_with_real_data.py _runs/m1_1000_20260118_083726_good\n")
    
    success = test_pairplot_with_real_data(run_dir)
    
    print("\n" + "=" * 60)
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. Please check the error messages above.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
