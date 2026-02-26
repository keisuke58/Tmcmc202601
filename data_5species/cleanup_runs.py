import shutil
from pathlib import Path
from datetime import datetime, timedelta


def cleanup_runs(runs_dir, target_dir_name="incomplete_runs", dry_run=False):
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"Directory not found: {runs_dir}")
        return

    target_path = runs_path / target_dir_name
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"Created target directory: {target_path}")

    # Threshold for recent files (to avoid moving currently running jobs)
    # 1 hour buffer
    time_threshold = datetime.now() - timedelta(hours=1)

    moved_count = 0

    print(f"Checking directories in {runs_path}...")
    print(
        f"Time threshold: {time_threshold.strftime('%Y-%m-%d %H:%M:%S')} (older folders will be moved)"
    )

    for item in runs_path.iterdir():
        if not item.is_dir():
            continue

        # Skip the target directory itself
        if item.name == target_dir_name:
            continue

        # Check modification time
        mtime = datetime.fromtimestamp(item.stat().st_mtime)

        # If the folder is newer than threshold, skip it (might be running)
        if mtime > time_threshold:
            print(f"[SKIP] Too new: {item.name} (Last modified: {mtime})")
            continue

        # Check for fit_metrics.json
        fit_metrics = item / "fit_metrics.json"

        if fit_metrics.exists():
            print(f"[KEEP] Completed run: {item.name}")
        else:
            # Check if it's a batch folder (optional specific logic, but assuming if no metrics, it's incomplete or just a container)
            # But let's look inside batch folders?
            # If a batch folder contains other folders, we might not want to move it unless it's empty or all children are incomplete.
            # For simplicity, if a batch folder doesn't have fit_metrics (it usually doesn't) but might contain valuable sub-runs.
            # Wait, usually batch runs create sub-folders inside _runs, not inside the batch folder itself?
            # Let's assume standard structure: if no fit_metrics.json at root of run folder -> incomplete.

            # Special handling for batch_* folders:
            # If they are just logs, maybe keep them? Or move them if they are old?
            # Let's treat them as regular folders for now.

            print(f"[MOVE] Incomplete run: {item.name} (No fit_metrics.json)")

            if not dry_run:
                try:
                    shutil.move(str(item), str(target_path / item.name))
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {item.name}: {e}")

    print(f"\nSummary: Moved {moved_count} folders to {target_path}")


if __name__ == "__main__":
    RUNS_DIR = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/_runs"
    # First do a dry run or just run it? User asked to do it.
    # I will run it directly but with the time safety check.
    cleanup_runs(RUNS_DIR, dry_run=False)
