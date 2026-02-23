# FEM Spatial Analysis

Spatial biofilm dynamics from 1D Hamilton+nutrient PDE through 3D tooth-scale FEM.

---

## 1D Spatial Dynamics

Hamilton ODE coupled with 1D nutrient diffusion (N=30 nodes, Monod kinetics).

### Spacetime Heatmaps — Commensal Static

![1D spacetime CS](images/1d_spacetime_commensal_static.png)

Space-time evolution of all 5 species across biofilm depth. Commensal species fill the domain uniformly; Pg remains negligible at all depths.

### Spacetime Heatmaps — Dysbiotic HOBIC (Baseline)

![1D spacetime DH](images/1d_spacetime_dh_baseline.png)

Dysbiotic condition shows Pg expansion starting near the saliva interface (x=1) where nutrient is abundant, with delayed penetration toward the tooth surface.

### Summary Panel — Commensal Static

![1D summary CS](images/1d_summary_commensal_static.png)

4-panel overview: spacetime, time series at selected depths, final composition, and DI profile.

### Summary Panel — Dysbiotic Baseline

![1D summary DH](images/1d_summary_dh_baseline.png)

Same layout for dysbiotic. Clear contrast in DI magnitude and spatial gradient.

---

## 3D Tooth-Scale FEM

Hamilton ODE → depth-dependent species composition → Open-Full-Jaw mesh (40k elements).

### 3D Cross-Section Slices — Dysbiotic HOBIC

![3D slices DH](images/3d_slices_dysbiotic_hobic.png)

Cross-sectional slices through the 3D tooth model showing species fields, DI, and nutrient concentration.

### 3D Cross-Section Slices — Commensal Static

![3D slices CS](images/3d_slices_commensal_static.png)

Same view for the commensal condition. Low DI throughout, uniform species composition.

### All 5 Species — Dysbiotic HOBIC (3D)

![3D all species DH](images/3d_all_species_dysbiotic_hobic.png)

Side-by-side 3D rendering of each species' volume fraction. Pg is concentrated near the outer surface; commensals dominate deeper layers.

### Pg Overview — All 4 Conditions

![Pg panel](images/panel_pg_overview_4conditions.png)

*P. gingivalis* spatial distribution across all conditions. Dysbiotic HOBIC (bottom-right) shows the highest Pg penetration.

### 3D Summary — All 4 Conditions

| Condition | Summary |
|-----------|---------|
| Commensal Static | ![3D summary CS](images/3d_summary_commensal_static.png) |
| Commensal HOBIC | ![3D summary CH](images/3d_summary_commensal_hobic.png) |
| Dysbiotic Static | ![3D summary DS](images/3d_summary_dysbiotic_static.png) |
| Dysbiotic HOBIC | ![3D summary DH](images/3d_summary_dysbiotic_hobic.png) |

---

## Dysbiotic Index Depth Profiles

### Cross-Condition DI Comparison

![DI cross condition](images/di_cross_condition.png)

DI depth profiles with 90% credible intervals across all conditions. DI is highest near the tooth surface for dysbiotic conditions.

### Pg Depth Comparison

![Pg depth](images/pg_depth_cross_condition.png)

*P. gingivalis* depth profiles. The Pg penetration gradient is a key discriminator between commensal and dysbiotic states.

### Posterior Pg Depth with CI

![Posterior Pg depth](images/posterior_depth_pg_comparison.png)

Cross-condition comparison of Pg depth profiles from TMCMC posterior samples.

### Stress-DI Joint Uncertainty

![Stress-DI](images/stress_di_uncertainty.png)

Joint distribution of von Mises stress and Dysbiotic Index, showing the coupling between mechanical load and microbial composition.

---

## DI Spatial Field — Dysbiotic HOBIC (3D)

![DI 3D DH](images/di_3d_dysbiotic_hobic.png)

Full 3D DI field on the tooth model. Higher DI (red) at the outer surface corresponds to Pg-dominated communities.

---

> See [Results Gallery](Results-Gallery) for the overview and key metrics.
> See [Multiscale Coupling](Multiscale-Coupling) for the 0D→1D→FEM pipeline details.
