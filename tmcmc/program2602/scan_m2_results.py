import os
import glob
import json
import numpy as np
import re
from pathlib import Path

# True parameters for M2
# a33, a34, a44, b3, b4
THETA_TRUE_M2 = np.array([1.5, 1.0, 2.0, 0.3, 0.4])


def parse_map_from_log(log_path):
    map_vec = None
    final_beta = 0.0
    is_m2 = False

    try:
        with open(log_path, "r") as f:
            # Read all lines is okay for these logs (usually < 10MB)
            # but let's be safe and read line by line or use a deque for last lines if files are huge.
            # Here we just read the whole file as it's easier to find context.
            content = f.read()

            # Check if it is M2
            if (
                "Model: M2" in content
                or '"model": "M2"' in content
                or "active_species: [2, 3]" in content
            ):
                is_m2 = True

            # Look for "TMCMC complete! Final β=1.0000"
            if "Final β=1.0000" in content:
                final_beta = 1.0

            # Look for Global MAP
            # 2026-01-29 22:00:08 INFO core.tmcmc: Global MAP: [1.38859152e+00 8.61163488e-04 1.01236082e+00 4.16946994e-04\n 1.79758781e-01]
            # Regex to capture the array. It might span multiple lines.

            # Strategy: Find the last occurrence of "Global MAP:"
            map_match = re.findall(r"Global MAP:\s*\[([\s\S]*?)\]", content)
            if map_match:
                last_map_str = map_match[-1]
                # Clean up newlines and extra spaces
                last_map_str = last_map_str.replace("\n", " ").strip()
                try:
                    map_vec = np.fromstring(last_map_str, sep=" ")
                except:
                    pass

    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None, 0.0, False

    return map_vec, final_beta, is_m2


def main():
    root_dir = "/home/nishioka/IKM_Hiwi/Tmcmc202601/tmcmc/tmcmc/_runs"
    log_files = glob.glob(os.path.join(root_dir, "**/run.log"), recursive=True)

    results = []

    print(f"Scanning {len(log_files)} log files...")

    for log_file in log_files:
        map_vec, final_beta, is_m2 = parse_map_from_log(log_file)

        if is_m2 and map_vec is not None and len(map_vec) == 5:
            # Calculate error
            diff = map_vec - THETA_TRUE_M2
            l2_error = np.linalg.norm(diff)
            rel_error = l2_error / np.linalg.norm(THETA_TRUE_M2)

            results.append(
                {
                    "path": os.path.dirname(log_file),
                    "map": map_vec,
                    "l2_error": l2_error,
                    "rel_error": rel_error,
                    "beta": final_beta,
                }
            )

    # Sort by error
    results.sort(key=lambda x: x["l2_error"])

    print(f"\nFound {len(results)} M2 runs with MAP estimates:\n")
    print(f"{'Run Directory':<60} | {'Status':<10} | {'L2 Error':<10} | {'MAP Estimate'}")
    print("-" * 120)

    for res in results:
        dir_name = os.path.basename(res["path"])
        status = "Done" if res["beta"] >= 0.99 else f"Beta={res['beta']:.2f}"
        map_str = np.array2string(res["map"], precision=3, separator=", ")
        print(f"{dir_name:<60} | {status:<10} | {res['l2_error']:.4f}     | {map_str}")


if __name__ == "__main__":
    main()
