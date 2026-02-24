#!/usr/bin/env bash
# ================================================================
# backup_to_gdrive.sh
# 重要データを _backup/ に集約し、Google Drive へ rclone sync
#
# Usage:
#   ./backup_to_gdrive.sh          # 集約 + sync
#   ./backup_to_gdrive.sh --local  # 集約のみ (sync しない)
#
# 初回セットアップ:
#   ~/.local/bin/rclone config
#   → "n" (new remote) → name: gdrive → type: drive
#   → client_id: (空Enter) → client_secret: (空Enter)
#   → scope: 1 (full access) → root_folder_id: (空Enter)
#   → service_account_file: (空Enter)
#   → Edit advanced config? n
#   → Use auto config? n → ブラウザでURL開いてコード貼る
#   → Shared drive? n → y (confirm)
# ================================================================
set -euo pipefail

PROJECT=/home/nishioka/IKM_Hiwi/Tmcmc202601
BACKUP=$PROJECT/_backup
REMOTE=gdrive:Tmcmc202601_backup   # rclone remote name : folder
RCLONE=$HOME/.local/bin/rclone
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Tmcmc202601 Backup — $TIMESTAMP ==="

# ---- 1. Create backup structure ----
mkdir -p "$BACKUP"/{posterior_samples,theta_MAP,paper_figures,prior_config,experiment_data,source_code,documents,wiki}

# ---- 2. Posterior samples (irreplaceable, ~90h compute each) ----
echo "[1/8] Posterior samples..."
# 5-species (4 conditions)
for d in "$PROJECT"/data_5species/_runs/*/; do
    cond=$(basename "$d")
    mkdir -p "$BACKUP/posterior_samples/5species/$cond"
    cp -u "$d"/*.npy "$BACKUP/posterior_samples/5species/$cond/" 2>/dev/null || true
    cp -u "$d"/*.json "$BACKUP/posterior_samples/5species/$cond/" 2>/dev/null || true
done
# dh_baseline main run
for d in "$PROJECT"/_runs/*/; do
    cond=$(basename "$d")
    mkdir -p "$BACKUP/posterior_samples/5species_sweeps/$cond"
    cp -u "$d"/*.npy "$BACKUP/posterior_samples/5species_sweeps/$cond/" 2>/dev/null || true
    cp -u "$d"/*.json "$BACKUP/posterior_samples/5species_sweeps/$cond/" 2>/dev/null || true
done
# Hill gate sweeps
for d in "$PROJECT"/_sweeps/*/; do
    cond=$(basename "$d")
    mkdir -p "$BACKUP/posterior_samples/sweeps/$cond"
    cp -u "$d"/*.npy "$BACKUP/posterior_samples/sweeps/$cond/" 2>/dev/null || true
    cp -u "$d"/*.json "$BACKUP/posterior_samples/sweeps/$cond/" 2>/dev/null || true
done
# M1/M2/M3 classical
for d in "$PROJECT"/tmcmc/tmcmc/_runs/*/; do
    cond=$(basename "$d")
    mkdir -p "$BACKUP/posterior_samples/classical_M1M2M3/$cond"
    cp -u "$d"/*.npy "$BACKUP/posterior_samples/classical_M1M2M3/$cond/" 2>/dev/null || true
    cp -u "$d"/*.json "$BACKUP/posterior_samples/classical_M1M2M3/$cond/" 2>/dev/null || true
    cp -u "$d"/*.npz "$BACKUP/posterior_samples/classical_M1M2M3/$cond/" 2>/dev/null || true
done

# ---- 3. theta_MAP (tiny but critical) ----
echo "[2/8] theta_MAP..."
find "$PROJECT" -name "theta_MAP*.json" -exec cp -u --parents {} "$BACKUP/theta_MAP/" \; 2>/dev/null || true

# ---- 4. Paper figures (Fig 8-16) ----
echo "[3/8] Paper figures..."
cp -ru "$PROJECT"/FEM/figures/paper_final/* "$BACKUP/paper_figures/" 2>/dev/null || true
cp -u "$PROJECT"/FEM/figures/3model_comparison_3d.png "$BACKUP/paper_figures/" 2>/dev/null || true

# ---- 5. Prior config ----
echo "[4/8] Prior config..."
cp -u "$PROJECT"/data_5species/model_config/*.json "$BACKUP/prior_config/" 2>/dev/null || true

# ---- 6. Experiment data ----
echo "[5/8] Experiment data..."
cp -ru "$PROJECT"/data_5species/experiment_data/*.png "$BACKUP/experiment_data/" 2>/dev/null || true
cp -ru "$PROJECT"/data_5species/experiment_data/*.csv "$BACKUP/experiment_data/" 2>/dev/null || true
cp -ru "$PROJECT"/data_5species/experiment_fig/*.png "$BACKUP/experiment_data/experiment_fig/" 2>/dev/null || true
cp -ru "$PROJECT"/data_5species/experiment_fig/*.jpg "$BACKUP/experiment_data/experiment_fig/" 2>/dev/null || true

# ---- 7. Key source code ----
echo "[6/8] Source code snapshot..."
# Core scripts only (not full git)
cp -u "$PROJECT"/data_5species/main/estimate_reduced_nishioka.py "$BACKUP/source_code/" 2>/dev/null || true
cp -u "$PROJECT"/data_5species/core/*.py "$BACKUP/source_code/" 2>/dev/null || true
cp -u "$PROJECT"/FEM/material_models.py "$BACKUP/source_code/" 2>/dev/null || true
cp -u "$PROJECT"/FEM/generate_paper_figures.py "$BACKUP/source_code/" 2>/dev/null || true
cp -u "$PROJECT"/FEM/plot_material_model_literature.py "$BACKUP/source_code/" 2>/dev/null || true
cp -u "$PROJECT"/FEM/plot_basin_sensitivity.py "$BACKUP/source_code/" 2>/dev/null || true

# ---- 8. Documents ----
echo "[7/8] Documents..."
cp -u "$PROJECT"/docs/*.pdf "$BACKUP/documents/" 2>/dev/null || true
cp -u "$PROJECT"/docs/*.tex "$BACKUP/documents/" 2>/dev/null || true
cp -u "$PROJECT"/PAPER_OUTLINE.md "$BACKUP/documents/" 2>/dev/null || true

# ---- 9. Wiki snapshot ----
echo "[8/8] Wiki..."
WIKI=/home/nishioka/IKM_Hiwi/Tmcmc202601.wiki
if [ -d "$WIKI" ]; then
    cp -ru "$WIKI"/*.md "$BACKUP/wiki/" 2>/dev/null || true
    mkdir -p "$BACKUP/wiki/images"
    cp -ru "$WIKI"/images/* "$BACKUP/wiki/images/" 2>/dev/null || true
fi

# ---- Summary ----
echo ""
echo "=== Backup summary ==="
du -sh "$BACKUP"/*/ 2>/dev/null | sort -h
echo "---"
du -sh "$BACKUP"
echo ""

# ---- 10. Sync to Google Drive ----
if [ "${1:-}" = "--local" ]; then
    echo "Local only (--local). Skipping Google Drive sync."
    exit 0
fi

if ! command -v "$RCLONE" &>/dev/null; then
    echo "ERROR: rclone not found at $RCLONE"
    echo "Install: curl https://rclone.org/install.sh | bash"
    exit 1
fi

if ! "$RCLONE" listremotes 2>/dev/null | grep -q "^gdrive:"; then
    echo "WARNING: rclone remote 'gdrive' not configured."
    echo "Run: $RCLONE config"
    echo "  → name: gdrive, type: drive, scope: 1 (full access)"
    echo "  → Google account: kei128608@gmail.com"
    exit 1
fi

echo "Syncing to $REMOTE ..."
"$RCLONE" sync "$BACKUP" "$REMOTE" \
    --progress \
    --exclude ".git/**" \
    --transfers 8 \
    --checkers 16

echo ""
echo "=== Done! Backup synced to Google Drive ==="
echo "  Local:  $BACKUP"
echo "  Remote: $REMOTE"
