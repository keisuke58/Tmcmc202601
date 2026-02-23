# Related work: LUH (IKM) biofilm modeling

## English paragraph (Markdown)

Biofilm growth has been extensively studied using continuum-mechanical approaches. Soleimani and co-workers developed finite-strain visco-elastic growth models driven by nutrient diffusion and applied them to biofilm growth, as well as to arterial pathologies such as dissection and atherosclerosis within a unified computational mechanics framework. Building on this line of research at the Institute of Continuum Mechanics (Leibniz Universität Hannover), Klempt et al. proposed a continuum multi-species biofilm model with a novel interaction scheme that can accommodate an arbitrary number of species and external agents such as antibiotics, demonstrating its ability to reproduce a wide range of biofilm interaction patterns. In parallel, Fritsch, Geisler and collaborators introduced a Bayesian model updating framework to calibrate biofilm constitutive parameters under hybrid uncertainties, combining epistemic uncertainty in model parameters with aleatory variability in biological responses. These studies emphasize continuum descriptions of biofilms and the identification of effective material laws. In contrast, the present work adopts a reduced multi-species population model with an explicit interaction matrix between oral bacterial groups, calibrated by Transitional Markov Chain Monte Carlo (TMCMC) against longitudinal in vitro data under commensal and dysbiotic, static and HOBIC conditions. This discrete-level representation sacrifices geometric detail but enables a direct ecological interpretation of inferred interaction strengths (competition, facilitation and bridge-organism effects) and a systematic comparison of how these interactions reorganize between health-associated and dysbiotic communities, while remaining computationally efficient enough to support extensive Bayesian inference and uncertainty quantification.

---

## LaTeX paragraph (for paper)

```latex
Biofilm growth has been extensively studied using continuum-mechanical
approaches. Soleimani and co-workers developed finite-strain
visco-elastic growth models driven by nutrient diffusion and applied
them to biofilm growth, as well as to arterial pathologies such as
dissection and atherosclerosis within a unified computational
mechanics framework~\cite{soleimani2019}. Building on this line of
research at the Institute of Continuum Mechanics (Leibniz Universität
Hannover), Klempt et al.\ proposed a continuum multi-species biofilm
model with a novel interaction scheme that can accommodate an
arbitrary number of species and external agents such as antibiotics,
demonstrating its ability to reproduce a wide range of biofilm
interaction patterns~\cite{klempt2025}. In parallel, Fritsch, Geisler
and collaborators introduced a Bayesian model updating framework to
calibrate biofilm constitutive parameters under hybrid uncertainties,
combining epistemic uncertainty in model parameters with aleatory
variability in biological responses~\cite{fritsch2025}. These studies
emphasize continuum descriptions of biofilms and the identification of
effective material laws. In contrast, the present work adopts a reduced
multi-species population model with an explicit interaction matrix
between oral bacterial groups, calibrated by Transitional Markov Chain
Monte Carlo (TMCMC) against longitudinal \emph{in vitro} data under
commensal and dysbiotic, static and HOBIC conditions. This discrete-level
representation sacrifices geometric detail but enables a direct
ecological interpretation of inferred interaction strengths
(competition, facilitation and bridge-organism effects) and a
systematic comparison of how these interactions reorganize between
health-associated and dysbiotic communities, while remaining
computationally efficient enough to support extensive Bayesian
inference and uncertainty quantification.
```

