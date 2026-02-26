import re
import json
from pathlib import Path

# Log files and their conditions
logs = {
    "Commensal_HOBIC": "Tmcmc202601/data_5species/main/commensal_hobic_4000.log",
    "Dysbiotic_HOBIC": "Tmcmc202601/data_5species/main/dysbiotic_hobic_4000.log",
    "Dysbiotic_Static": "Tmcmc202601/data_5species/main/dysbiotic_static_4000.log",
}

# Output directories (from previous grep)
output_dirs = {
    "Commensal_HOBIC": "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/Commensal_HOBIC_20260207_023055",
    "Dysbiotic_HOBIC": "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/Dysbiotic_HOBIC_20260207_023107",
    "Dysbiotic_Static": "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs/Dysbiotic_Static_20260207_022019",
}


def extract_map_vector(log_path):
    with open(log_path, "r") as f:
        content = f.read()

    # Regex to find the MAP vector
    # Pattern: �� MAP (posterior (observation-corrected)): [ ... ]
    match = re.search(
        r"🎯 MAP \(posterior \(observation-corrected\)\): \[(.*?)\]", content, re.DOTALL
    )
    if match:
        vector_str = match.group(1).replace("\n", " ")
        # Remove multiple spaces
        vector_str = re.sub(r"\s+", " ", vector_str).strip()
        try:
            return [float(x) for x in vector_str.split()]
        except ValueError:
            print(f"Error parsing vector in {log_path}")
            return None
    return None


def save_theta_map(condition, vector):
    if not vector:
        print(f"Skipping {condition}: No vector found")
        return

    out_dir = Path(output_dirs[condition])
    if not out_dir.exists():
        print(f"Skipping {condition}: Output dir not found {out_dir}")
        return

    # Create dummy theta_MAP.json
    data = {
        "theta_sub": vector,
        "theta_full": vector,  # Assuming full estimation for now
        "active_indices": list(range(20)),
        "note": "Extracted from Chain 1 logs",
    }

    out_path = out_dir / "theta_MAP.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {out_path}")


for cond, log_file in logs.items():
    print(f"Processing {cond}...")
    vector = extract_map_vector(log_file)
    if vector:
        print(f"  Vector found (len={len(vector)})")
        save_theta_map(cond, vector)
    else:
        print("  No vector found")
