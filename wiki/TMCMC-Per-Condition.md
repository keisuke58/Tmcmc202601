# TMCMC Results — Per-Condition Detail

Detailed posterior inference results for each of the 4 experimental conditions (1000 particles, ~90 h per run).

---

## Commensal Static

Negative control: no dysbiotic inoculum, static medium.

### MAP Fit

![MAP fit CS](images/map_fit_commensal_static.png)

Pg remains suppressed throughout. a₃₅ and a₄₅ near-zero confirm absence of bridge-mediated facilitation.

### Posterior Predictive Band

![Posterior band CS](images/posterior_band_commensal_static.png)

90% credible interval. Tight bands for dominant species, reflecting well-constrained growth parameters.

---

## Commensal HOBIC

Commensal inoculum under dynamic HOBIC flow.

### MAP Fit

![MAP fit CH](images/map_fit_commensal_hobic.png)

HOBIC flow alters growth dynamics but Pg remains low. Commensal species composition is maintained.

### Posterior Predictive Band

![Posterior band CH](images/posterior_band_commensal_hobic.png)

Credible bands slightly wider than static due to flow-induced variability.

### Species Composition

![Composition CH](images/composition_commensal_hobic.png)

Time evolution of relative abundance. *S. oralis* and *A. naeslundii* dominate throughout.

### Parameter Violins

![Violins CH](images/violins_commensal_hobic.png)

Posterior distributions for all 20 parameters. Growth rates (r₁–r₅) and carrying capacities (K₁–K₅) are well-identified.

### Correlation Matrix

![Correlation CH](images/correlation_commensal_hobic.png)

Parameter correlations. Note the strong r–K anticorrelation (classical in logistic growth models).

### Corner Plot

![Corner CH](images/corner_commensal_hobic.png)

Full pairwise posterior scatter. Unimodal posteriors confirm convergence.

---

## Dysbiotic Static

Dysbiotic inoculum (elevated Fn + Pg), static medium.

### MAP Fit

![MAP fit DS](images/map_fit_dysbiotic_static.png)

Pg surge begins at ~100 h, driven by bridge organism facilitation. The model captures the delayed onset well.

### Posterior Predictive Band

![Posterior band DS](images/posterior_band_dysbiotic_static.png)

Wider uncertainty for Pg and Fn reflects the nonlinear amplification of bridge interactions.

### Interaction Heatmap

![Heatmap DS](images/interaction_heatmap_dysbiotic_static.png)

Inferred interaction matrix. a₃₅ (Vd→Pg) and a₄₅ (Fn→Pg) are large and positive — key dysbiotic drivers.

### Species Composition

![Composition DS](images/composition_dysbiotic_static.png)

Relative abundance shift: commensals decline as Pg expands after ~120 h.

### Parameter Violins

![Violins DS](images/violins_dysbiotic_static.png)

Bridge interaction parameters (a₃₅, a₄₅) are strongly identified. Commensal growth rates remain similar to the commensal conditions.

### Correlation Matrix

![Correlation DS](images/correlation_dysbiotic_static.png)

a₃₅ and a₄₅ show positive correlation — consistent with co-facilitation of Pg by bridge organisms.

### Sensitivity Analysis

![Sensitivity DS](images/sensitivity_dysbiotic_static.png)

Local sensitivity of model output to each parameter. Pg dynamics are most sensitive to a₃₅ and a₄₅.

---

## Dysbiotic HOBIC

Dysbiotic inoculum under dynamic HOBIC flow — the most clinically relevant condition.

### MAP Fit

![MAP fit DH](images/map_fit_dysbiotic_hobic.png)

The strongest Pg surge, well-captured by the model. Bridge-mediated facilitation is amplified under flow.

### Posterior Predictive Band

![Posterior band DH](images/posterior_band_dysbiotic_hobic.png)

Widest credible intervals among all conditions, reflecting the largest posterior uncertainty in bridge parameters.

### Interaction Heatmap

![Heatmap DH](images/interaction_heatmap_dysbiotic_hobic.png)

The largest a₃₅ and a₄₅ values across all conditions.

### Parameter Violins

![Violins DH](images/violins_dysbiotic_hobic.png)

Bridge parameters are strongly constrained by the pronounced Pg surge data.

### Correlation Matrix

![Correlation DH](images/correlation_dysbiotic_hobic.png)

Strong a₃₅–a₄₅ correlation persists under flow conditions.

### Sensitivity Analysis

![Sensitivity DH](images/sensitivity_dysbiotic_hobic.png)

Pg output sensitivity dominated by bridge interactions, with secondary dependence on Hill gate parameters.

---

## Cross-Condition Comparison

### Interaction Network

![Network](images/interaction_network.png)

Summary network showing facilitation (blue) and inhibition (red). Bridge-mediated Pg facilitation is the dominant feature distinguishing dysbiotic from commensal conditions.

---

> See [Results Gallery](Results-Gallery) for the overview table and key highlights.
