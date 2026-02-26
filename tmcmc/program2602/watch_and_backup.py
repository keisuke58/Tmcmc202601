import time
import shutil
from pathlib import Path
import os

BASE_DIR = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/_runs/debug_M4_M5_100p")
REFINED_DIR = BASE_DIR / "refined"
M4_JSON = BASE_DIR / "theta_MAP_M4_refined.json"
M4_TRACE_DEST = REFINED_DIR / "trace_M4_refined.npy"
TRACE_SRC = REFINED_DIR / "trace_refined.npy"

print(f"Watching for {M4_JSON}...")

while True:
    if M4_JSON.exists():
        if not M4_TRACE_DEST.exists():
            if TRACE_SRC.exists():
                print(f"M4 finished! Copying {TRACE_SRC} to {M4_TRACE_DEST}")
                try:
                    shutil.copy(TRACE_SRC, M4_TRACE_DEST)
                    print("Backup successful.")
                except Exception as e:
                    print(f"Backup failed: {e}")
            else:
                print(f"Warning: {TRACE_SRC} not found when M4 finished.")
        else:
            # print("M4 trace already backed up.")
            pass
        break

    time.sleep(1)

print("Watcher finished.")
