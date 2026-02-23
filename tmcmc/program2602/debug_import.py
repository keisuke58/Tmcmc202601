import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(".").resolve()))
print(f"DEBUG: sys.path: {sys.path}")

print("Importing config...")
try:
    import config
    print("Config imported successfully.")
    print(f"TMCMC_DEFAULTS: {config.TMCMC_DEFAULTS}")
except Exception as e:
    print(f"Error importing config: {e}")
    import traceback
    traceback.print_exc()
