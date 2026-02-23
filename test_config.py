import sys
import os
sys.path.append('/home/nishioka/IKM_Hiwi/Tmcmc202601')
sys.path.append('/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/program2602') # For config.py
from data_5species.core.nishioka_model import get_condition_bounds

def test():
    conditions = [
        ("Commensal", "Static"),
        ("Dysbiotic", "HOBIC"),
        ("Dysbiotic", "Static"),
        ("Commensal", "HOBIC")
    ]
    
    for cond, cult in conditions:
        print(f"Testing {cond} {cult}...")
        try:
            bounds, locks = get_condition_bounds(cond, cult)
            print(f"  Locks count: {len(locks)}")
            # Print a few key bounds to verify
            if cond == "Commensal" and cult == "Static":
                print(f"  b1(3): {bounds[3]}") # Should be (0.0, 3.0)
                print(f"  b4(9): {bounds[9]}") # Should be (0.0, 0.0) - locked
                
            if cond == "Dysbiotic" and cult == "HOBIC":
                print(f"  b3(8): {bounds[8]}") # Should be (0.0, 5.0)
                print(f"  a35(18): {bounds[18]}") # Should be (-3.0, 0.0)

            if cond == "Dysbiotic" and cult == "Static":
                print(f"  a33(5): {bounds[5]}") # Should be (1.0, 5.0)
                print(f"  b5(15): {bounds[15]}") # Should be (0.0, 0.2)
                
            if cond == "Commensal" and cult == "HOBIC":
                print(f"  b1(3): {bounds[3]}") # Should be (1.0, 10.0)
                print(f"  b5(15): {bounds[15]}") # Should be (0.0, 0.0) locked
                
        except Exception as e:
            print(f"  ERROR: {e}")
            
        print("-" * 20)

if __name__ == "__main__":
    test()
