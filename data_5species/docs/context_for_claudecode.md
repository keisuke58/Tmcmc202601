# Current Project Status & Implementation Summary (2026-02-05)

## 1. Project Goal
Reproduce the biological findings of Heine et al. (2025) using a 5-species biofilm model estimated via TMCMC.
Focus: Validating the "Surge" mechanism in Dysbiotic HOBIC conditions while ensuring suppression in Negative Controls.

## 2. Current Execution Status
Calculations are currently running on 4 remote servers (started ~2026-02-05).

| Condition | Cultivation | Server | Status | Expected Outcome (Validation) |
|---|---|---|---|---|
| **Commensal** | **Static** | `marinos03` | Running | **Negative Control**: S4/S5 suppressed (Abundance ≤ 1%) |
| **Dysbiotic** | **Static** | `frontale04` | Running | **Negative Control**: S4/S5 suppressed (Abundance ≤ 1%) |
| **Commensal** | **HOBIC** | `frontale02` | Running | **Negative Control**: S4/S5 suppressed (Abundance ≤ 1%) |
| **Dysbiotic** | **HOBIC** | `marinos02` | Running | **Target**: "Surge" of S5 (Red) + S4 (Purple) |

## 3. Validation Logic (Crucial for Analysis)
We have implemented an automated validation script `verify_biological_validity.py`.
Any AI assistant analyzing the results MUST understand the following **Parameter Logic**:

### A. Symmetric Matrix Assumption
The model assumes a symmetric interaction matrix ($a_{ij} = a_{ji}$).
- **Lactate Handover (S1->S3)**: Normally $a_{31}$, but in the solver vector `theta`, this is stored as **`a13`** (Index 10).
- **pH Trigger (S3->S5)**: Normally $a_{53}$, but in the solver vector `theta`, this is stored as **`a35`** (Index 18).

### B. Signatures of Success (Dysbiotic HOBIC)
To confirm the model captures the biological mechanism, the estimated parameters must satisfy:
1.  **Lactate Handover**: `a13 < 0` (S1 helps S3).
2.  **pH Trigger**: `a35 < -0.5` (Strong cooperation from S3 to S5).
3.  **Surge Curve**: Simulation must show S5 abundance increasing significantly from Day 10 to Day 21.

## 4. File Structure & Tools
- `data_5species/main/verify_biological_validity.py`: Checks the above signatures automatically.
- `data_5species/main/sync_results.sh`: Downloads results from all servers to `_runs/`.
- `data_5species/docs/next_plan20260205`: Original roadmap.

## 5. Next Actions (Post-Calculation)
1.  Run `sync_results.sh` to collect data.
2.  Run `verify_biological_validity.py` on the `Dysbiotic_HOBIC` result.
3.  **If Valid**: Proceed to plotting (Paper Figure reproduction).
4.  **If Invalid**: Re-examine parameter bounds for `a13` and `a35`.
