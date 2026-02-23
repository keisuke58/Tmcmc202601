
import sys
import os
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601")
DATA_5SPECIES_ROOT = PROJECT_ROOT / "data_5species"
sys.path.insert(0, str(DATA_5SPECIES_ROOT))

from core.nishioka_model import get_condition_bounds

def test_bounds():
    condition = "Dysbiotic"
    cultivation = "HOBIC"
    
    print(f"Testing bounds for {condition} {cultivation}...")
    bounds, locked = get_condition_bounds(condition, cultivation)
    
    print("Locked indices:", locked)
    
    # Check specific indices for Surge (18: a35, 19: a45)
    print(f"Theta[18] (a35 S3->S5) bound: {bounds[18]}")
    print(f"Theta[19] (a45 S4->S5) bound: {bounds[19]}")
    
    # Check if they allow negative values
    if bounds[18][1] <= 0:
        print("PASS: Theta[18] allows/forces negative values (Cooperation).")
    else:
        print("FAIL: Theta[18] allows positive values (Competition).")

if __name__ == "__main__":
    test_bounds()
