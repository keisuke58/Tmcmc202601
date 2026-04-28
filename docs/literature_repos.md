# 文献調査: Clone候補リポジトリ & 関連論文 (2026-03-11)

## 1. VEM (Virtual Element Method)

### Clone候補
| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 61 | [Terenceyuyue/mVEM](https://github.com/Terenceyuyue/mVEM) | MATLAB | VEM教育実装 (Poisson, elasticity, Stokes) |
| 40 | [CAMLAB-UChile/vemlab](https://github.com/CAMLAB-UChile/vemlab) | MATLAB | 2D VEM for Poisson/elasticity/Stokes (Ortiz-Bernardin) |
| 20 | [justachetan/VirtualElementMethods](https://github.com/justachetan/VirtualElementMethods) | Python/Jupyter | "VEM in 50 lines" Python実装 |
| 20 | [cemcen/Veamy](https://github.com/cemcen/Veamy) | C++ | 2D VEM + Voronoi mesh |
| 14 | [Qinxiaoye/VEM-spcae-time-dynamic](https://github.com/Qinxiaoye/VEM-spcae-time-dynamic) | MATLAB | **Space-time VEM** — 動的解析 |
| 14 | [tytrusty/Shape-Matching-Element-Method](https://github.com/tytrusty/Shape-Matching-Element-Method) | C | Shape matching VEM for CAD |
| 7 | [MinhNguyenIKM/vem](https://github.com/MinhNguyenIKM/vem) | MATLAB | **IKM出身**のVEM実装！ |
| 7 | [nsundar/VEM_in_Abaqus](https://github.com/nsundar/VEM_in_Abaqus) | Fortran | VEM in Abaqus (UEL) |
| 7 | [deatinor/VEM3D](https://github.com/deatinor/VEM3D) | C++ | 3D VEM実装 |
| 6 | [nickdisca/VEM_project](https://github.com/nickdisca/VEM_project) | C++ | 2D advection-diffusion VEM |
| 3 | [aerubianoma/vem_stress_assisted_diffusion](https://github.com/aerubianoma/vem_stress_assisted_diffusion) | MATLAB | **Stress-assisted diffusion VEM** — chemo-mech結合に直結 |
| 3 | [Paulms/VEMPoisson2D](https://github.com/Paulms/VEMPoisson2D) | Julia | Julia VEM (Poisson) |

### 特に注目
- **MinhNguyenIKM/vem** — IKM (Leibniz Universität Hannover) 出身、我々の研究室のコード
- **vem_stress_assisted_diffusion** — 応力駆動拡散、バイオフィルムの chemo-mech 結合に直接関連
- **VEM-spcae-time-dynamic** — Space-Time VEM、我々の `vem_spacetime.py` と比較可能

### 主要論文
- Beirao da Veiga et al. (2013) "Basic principles of VEM" — 基礎論文
- Anaya et al. (2020) "VEM for FitzHugh-Nagumo" — 生体反応拡散にVEM適用
- Artioli et al. (2022) "Adaptive curved VEM for composites" — heterogeneous material
- Hussein et al. (2019-2022) "VEM for crack propagation" — 界面追跡 (biofilm detachment に関連)

---

## 2. バイオフィルム力学

### Clone候補
| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 22 | [nufeb/NUFEB](https://github.com/nufeb/NUFEB) | C++ | LAMMPS-based バイオフィルムシミュレータ (Newcastle) |
| 18 | [kreft/iDynoMiCS](https://github.com/kreft/iDynoMiCS) | Java | 最多引用の agent-based biofilm simulator (Birmingham) |
| 11 | [kreft/iDynoMiCS-2](https://github.com/kreft/iDynoMiCS-2) | Java | 次世代 iDynoMiCS (deformable cells) |
| 7 | [nufeb/NUFEB-2](https://github.com/nufeb/NUFEB-2) | C++ | NUFEB v2 |
| 1 | [f-chenyi/biofilm-mechanics-theory](https://github.com/f-chenyi/biofilm-mechanics-theory) | Python | バイオフィルム力学理論 |
| 1 | [ealopez/Biofilm-mechanics](https://github.com/ealopez/Biofilm-mechanics) | Python | バイオフィルム力学 |

### 主要論文
- **Horvat et al. (2023)** "Mechanical properties of biofilms — A review" Biotechnol. Bioeng. — E値の包括レビュー
- **Gloag et al. (2020)** "Biofilm mechanics: implications in infection" Biofilm 2:100017
- **Kovach et al. (2020)** "Evolutionary adaptations of biofilms..." npj Biofilms & Microbiomes — EPS組成→力学
- **Seminara et al. (2012)** "Osmotic spreading of B. subtilis biofilms" PNAS — poroelastic model
- **Yan et al. (2019)** "Mechanical instability and interfacial energy" eLife — EPS production → E
- **Tierra et al. (2015)** "Multicomponent model of deformation and detachment" J. R. Soc. Interface — FEM + hyperelastic biofilm
- **Klapper & Dockery (2010)** "Mathematical description of microbial biofilms" SIAM Review — 標準的な連続体枠組み

---

## 3. TMCMC / Bayesian Inference

### Clone候補
| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 346 | [SURGroup/UQpy](https://github.com/SURGroup/UQpy) | Python | UQ framework — **TMCMCを直接実装** |
| 799 | [sbi-dev/sbi](https://github.com/sbi-dev/sbi) | Python | Simulation-Based Inference (neural posterior estimation) |
| 643 | [bayesflow-org/bayesflow](https://github.com/bayesflow-org/bayesflow) | Python | Deep learning + Bayesian modeling |
| 167 | [undark-lab/swyft](https://github.com/undark-lab/swyft) | Python | SBI at scale |
| 23 | [mukeshramancha/transitional-mcmc](https://github.com/mukeshramancha/transitional-mcmc) | Python | Pure Python TMCMC |
| 17 | [AnderGray/TransitionalMCMC.jl](https://github.com/AnderGray/TransitionalMCMC.jl) | Julia | Julia TMCMC |
| 15 | [diegoandresalvarez/BWBN_TMCMC](https://github.com/diegoandresalvarez/BWBN_TMCMC) | MATLAB | Bouc-Wen model + TMCMC |
| 7 | [cselab/old_korali](https://github.com/cselab/old_korali) | C++ | ETH Zurich Korali (TMCMC + optimization) |
| 6 | [amaya-macarena/ASMC](https://github.com/amaya-macarena/ASMC) | Python | Adaptive SMC for Bayesian inference |
| 4 | [PSengupta623/Affine-invariance-TMCMC](https://github.com/PSengupta623/Affine-invariance-TMCMC) | MATLAB | Affine-invariant TMCMC |
| 2 | [gtarabat/smTMCMC](https://github.com/gtarabat/smTMCMC) | MATLAB | Simplified manifold TMCMC |
| 0 | [dmkolovos/Bayesian-Inference-Methods-in-Structural-Dynamics](https://github.com/dmkolovos/Bayesian-Inference-Methods-in-Structural-Dynamics) | Python | MH, HMC, TMCMC for structural dynamics |

### 特に注目
- **UQpy** — TMCMC直接実装あり、ベンチマーク比較に最適
- **sbi** — DeepONet-TMCMC パイプラインの代替手法比較
- **Korali (old)** — ETH Zurich、元祖TMCMC最適化フレームワーク
- **transitional-mcmc** — シンプルなPython実装、アルゴリズム比較用

### 主要論文
- **Ching & Chen (2007)** "Transitional MCMC" J. Eng. Mech. — 原論文
- **Wu et al. (2018)** "Parallel adaptive TMCMC" SIAM J Sci Comput — Korali
- **Dau & Chopin (2022)** "Waste-free SMC" JRSS-B — 効率的SMC
- **Cranmer et al. (2020)** "Frontier of simulation-based inference" PNAS — SBI overview
- **Stuart (2010)** "Inverse problems: a Bayesian perspective" Acta Numerica — 基礎
- **Peirlinck, Kuhl et al. (2019-2024)** Bayesian calibration of tissue mechanics (Stanford)

---

## 4. Multiscale Coupling

### Clone候補
| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 2 | [mathLab/MorphoelasticRod](https://github.com/mathLab/MorphoelasticRod) | Python | 形態弾性 rod model (成長+力学) |
| 2 | [karinaurazova/growth_artery_with_residual_stress](https://github.com/karinaurazova/growth_artery_with_residual_stress) | Python/Jupyter | 動脈の成長+残留応力 |
| 0 | [axelalmet/MorphoelasticCrypt](https://github.com/axelalmet/MorphoelasticCrypt) | R | 腸クリプトの形態弾性 |
| 0 | [Mar5bar/morphoelastic-tumour](https://github.com/Mar5bar/morphoelastic-tumour) | MATLAB | 腫瘍成長の形態弾性 |

### 主要論文・フレームワーク (training knowledge)
- **FEniCS**: `dolfin-adjoint` for PDE-constrained optimization + growth
- **preCICE** (github.com/precice/precice, ~900⭐): multi-physics coupling library (FEM↔ODE↔CFD)
- **MuPhyN/FEBio** (github.com/febiosoftware/FEBio): biomechanics FEM, tissue growth
- **Klempt (2024)**: staggered coupling (our reference implementation)
- **Ambrosi & Mollica (2002)**: "On the mechanics of a growing tumor" — continuum growth mechanics
- **Rodriguez et al. (1994)**: multiplicative decomposition F = F_e · F_g — growth kinematics
- **Goriely (2017)**: "Mathematics and Mechanics of Biological Growth" — textbook

---

## 優先Clone推奨 (Top 10)

| # | Repo | 理由 |
|---|------|------|
| 1 | `SURGroup/UQpy` | TMCMC直接実装、ベンチマーク比較 |
| 2 | `sbi-dev/sbi` | SBI手法比較 (DeepONet代替) |
| 3 | `Terenceyuyue/mVEM` | VEM教育コード (MATLAB → Python移植参考) |
| 4 | `CAMLAB-UChile/vemlab` | 2D VEM参照実装 |
| 5 | `MinhNguyenIKM/vem` | IKM出身、研究室つながり |
| 6 | `nufeb/NUFEB` | バイオフィルムシミュレーション比較 |
| 7 | `kreft/iDynoMiCS` | 最多引用 agent-based biofilm |
| 8 | `mukeshramancha/transitional-mcmc` | シンプルTMCMC比較 |
| 9 | `aerubianoma/vem_stress_assisted_diffusion` | VEM + chemo-mech |
| 10 | `justachetan/VirtualElementMethods` | Python VEM (JAX移植ベース) |

---

## 追加Clone (Multiscale + Biofilm解析)

| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 901 | [precice/precice](https://github.com/precice/precice) | C++ | Partitioned multi-physics coupling library (※大規模、参照のみ) |
| 608 | [kinnala/scikit-fem](https://github.com/kinnala/scikit-fem) | Python | 軽量FEM (reaction-diffusion + elasticity 例あり) |
| 249 | [febiosoftware/FEBio](https://github.com/febiosoftware/FEBio) | C++ | 生体力学FEM (growth, remodeling, biphasic, SLS) |
| 71 | [cellmodeller/CellModeller](https://github.com/cellmodeller/CellModeller) | Python | GPU agent-based biofilm (OpenCL) |
| 10 | [knutdrescher/BiofilmQ](https://github.com/knutdrescher/BiofilmQ) | MATLAB | バイオフィルム画像解析 |
| 0 | [BAMresearch/FenicsXConcrete](https://github.com/BAMresearch/FenicsXConcrete) | Python | FEniCSx: コンクリート aging kinetics + FEM staggered coupling |

### Multiscale 追加論文
- **preCICE**: Bungartz et al. (2022) "preCICE v2" Open Research Europe
- **FEBio**: Maas et al. (2012) "FEBio" J Biomech Eng — 生体力学の標準ソフト
- **Penta & Gerisch (2020)** "Asymptotic homogenization for tumour growth" Bull Math Biol — adiabatic scale separation
- **Tepole et al. (2023)** "Chemo-bio-mechanical coupling" CMAME — growth tensor approach
- **Tierra & Guillen-Gonzalez (2021)** "Phase-field biofilm" CMAME — Cahn-Hilliard + mechanics

---

## 5. DeepONet / Surrogate Modeling

### Clone候補
| ⭐ | Repo | Lang | 概要 |
|----|------|------|------|
| 3945 | [lululxvi/deepxde](https://github.com/lululxvi/deepxde) | Python | **DeepXDE**: PINN + DeepONet統合ライブラリ (Lu Lu) |
| 3454 | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | Python | **公式 Neural Operator**: FNO, TFNO, UNO, etc. (Caltech/Anima) |
| 767 | [lululxvi/deeponet](https://github.com/lululxvi/deeponet) | Python | **DeepONet 原論文コード** (Lu Lu, Brown→Yale) |
| 393 | [PredictiveIntelligenceLab/Physics-informed-DeepONets](https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets) | Jupyter | **PI-DeepONet** (Karniadakis lab) |
| 361 | [lu-group/deeponet-fno](https://github.com/lu-group/deeponet-fno) | Python | **DeepONet vs FNO 公正比較** — ベンチマーク必読 |
| 308 | [neuraloperator/Geo-FNO](https://github.com/neuraloperator/Geo-FNO) | Jupyter | **Geometry-Aware FNO** — 非規則メッシュ対応 |
| 79 | [katiana22/latent-deeponet](https://github.com/katiana22/latent-deeponet) | Python | **Latent DeepONet** — 潜在空間でのoperator学習 |
| 39 | [HewlettPackard/separable-operator-networks](https://github.com/HewlettPackard/separable-operator-networks) | Python | **SepONet** — 超大規模operator学習 |
| 3 | [amaya-macarena/ASMC-SURR](https://github.com/amaya-macarena/ASMC-SURR) | Python | **Adaptive SMC + Surrogate** — サロゲート加速Bayesian |

### 特に注目
- **deepxde** — DeepONet実装の標準ライブラリ、我々のカスタム実装と比較可能
- **deeponet-fno** — DeepONet vs FNO ベンチマーク、論文で引用すべき
- **latent-deeponet** — 潜在空間DeepONet、高次元出力の効率化に有望
- **ASMC-SURR** — サロゲート+SMC、DeepONet-TMCMCパイプラインの代替手法
- **Geo-FNO** — 非規則メッシュ上のFNO、conformal mesh対応の可能性

### 主要論文
- **Lu et al. (2021)** "Learning nonlinear operators via DeepONet" *Nature Machine Intelligence* — 原論文
- **Lu et al. (2022)** "A comprehensive and fair comparison of DeepONet and FNO" arXiv — deeponet-fno repo
- **Li et al. (2021)** "Fourier Neural Operator" *ICLR* — FNO原論文
- **Kontolati et al. (2024)** "Latent DeepONet for real-time predictions" *Nature Comp. Sci.*
- **Cranmer et al. (2020)** "Frontier of simulation-based inference" *PNAS*
- **Wang et al. (2021)** "Long-time integration of PI-DeepONets" — 長時間積分の安定性
- **Kissas et al. (2022)** "Learning operators with coupled attention" — Transformer-based operator
- **Goswami et al. (2023)** "DeepONet for fracture mechanics" — 構造力学への応用例

### サロゲート × Bayesian 推論
- **ASMC-SURR (Amaya et al.)** — Adaptive SMC with forward surrogate
- **Cranmer (2020) SBI** — Neural posterior estimation
- **Linka & Kuhl (2023)** "Bayesian physics-informed neural networks for tissue mechanics" — PINN × Bayesian
