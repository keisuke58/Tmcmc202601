#!/bin/bash
# 文献調査 Clone スクリプト (2026-03-11)
# Usage: bash docs/clone_repos.sh [category]
#   category: vem, biofilm, tmcmc, multiscale, all (default: all)

BASEDIR="$HOME/IKM_Hiwi/external_repos"
mkdir -p "$BASEDIR"

category="${1:-all}"

clone_if_missing() {
    local repo="$1"
    local dir="$BASEDIR/$(basename "$repo")"
    if [ -d "$dir" ]; then
        echo "  [SKIP] $repo already exists"
    else
        echo "  [CLONE] $repo"
        git clone --depth 1 "https://github.com/$repo.git" "$dir"
    fi
}

# ── VEM ──
if [ "$category" = "vem" ] || [ "$category" = "all" ]; then
    echo "=== VEM ==="
    clone_if_missing "Terenceyuyue/mVEM"
    clone_if_missing "CAMLAB-UChile/vemlab"
    clone_if_missing "justachetan/VirtualElementMethods"
    clone_if_missing "MinhNguyenIKM/vem"
    clone_if_missing "aerubianoma/vem_stress_assisted_diffusion"
    clone_if_missing "Qinxiaoye/VEM-spcae-time-dynamic"
    clone_if_missing "deatinor/VEM3D"
    clone_if_missing "nsundar/VEM_in_Abaqus"
fi

# ── Biofilm ──
if [ "$category" = "biofilm" ] || [ "$category" = "all" ]; then
    echo "=== Biofilm ==="
    clone_if_missing "nufeb/NUFEB"
    clone_if_missing "kreft/iDynoMiCS"
    clone_if_missing "kreft/iDynoMiCS-2"
    clone_if_missing "f-chenyi/biofilm-mechanics-theory"
fi

# ── TMCMC / Bayesian ──
if [ "$category" = "tmcmc" ] || [ "$category" = "all" ]; then
    echo "=== TMCMC / Bayesian ==="
    clone_if_missing "SURGroup/UQpy"
    clone_if_missing "sbi-dev/sbi"
    clone_if_missing "mukeshramancha/transitional-mcmc"
    clone_if_missing "AnderGray/TransitionalMCMC.jl"
    clone_if_missing "cselab/old_korali"
    clone_if_missing "amaya-macarena/ASMC"
    clone_if_missing "dmkolovos/Bayesian-Inference-Methods-in-Structural-Dynamics"
fi

# ── Multiscale ──
if [ "$category" = "multiscale" ] || [ "$category" = "all" ]; then
    echo "=== Multiscale ==="
    clone_if_missing "mathLab/MorphoelasticRod"
    clone_if_missing "karinaurazova/growth_artery_with_residual_stress"
    clone_if_missing "bayesflow-org/bayesflow"
    clone_if_missing "kinnala/scikit-fem"
    clone_if_missing "febiosoftware/FEBio"
    clone_if_missing "BAMresearch/FenicsXConcrete"
fi

# ── Biofilm 追加 ──
if [ "$category" = "biofilm" ] || [ "$category" = "all" ]; then
    clone_if_missing "cellmodeller/CellModeller"
    clone_if_missing "knutdrescher/BiofilmQ"
fi

# ── DeepONet / Surrogate ──
if [ "$category" = "surrogate" ] || [ "$category" = "all" ]; then
    echo "=== DeepONet / Surrogate ==="
    clone_if_missing "lululxvi/deepxde"
    clone_if_missing "lululxvi/deeponet"
    clone_if_missing "neuraloperator/neuraloperator"
    clone_if_missing "PredictiveIntelligenceLab/Physics-informed-DeepONets"
    clone_if_missing "lu-group/deeponet-fno"
    clone_if_missing "katiana22/latent-deeponet"
    clone_if_missing "neuraloperator/Geo-FNO"
    clone_if_missing "HewlettPackard/separable-operator-networks"
    clone_if_missing "amaya-macarena/ASMC-SURR"
fi

echo ""
echo "Done. Repos cloned to: $BASEDIR"
ls -1 "$BASEDIR" 2>/dev/null | wc -l
echo "repos total."
