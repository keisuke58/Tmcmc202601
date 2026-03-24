## Felix monospecies: dynamic nutrient parameter set

This memo stores the parameter values you provided for the Felix formulation of the monospecies Hamilton ODE.

### Dynamic nutrient term

`c(t) = 10 * (1 - phi*psi)`

### Temperature-dependent parameter

`a11(T)`:

| Temp (C) | a11 |
|---:|---:|
| 4  | 4.25 |
| 8  | 10 |
| 15 | 25 |
| 20 | 50 |
| 25 | 100 |
| 35 | 110 |
| 37 | 115 |
| 40 | 40 |

### Other fixed parameters

- `Kp = 1e-4`
- `eta = 1`
- `eta_phi = 1`
- `alpha = 0`
- `b11 = 0`
- `dt = 1e-4`

EOF
