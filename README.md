# Sparse and Bayesian Linear Regression from Scratch

Rebuild the linear-regression toolkit of Murphy's Probabilistic Machine Learning (chapter 11) in NumPy and reproduce its results. Solve least squares by normal equations, QR back-substitution and the SVD, and break the normal equations on an ill-conditioned design. Fit ridge regression through virtual data and the SVD, measure its effective degrees of freedom, and pick its strength by the evidence and by cross-validation. Write the lasso shooting algorithm, trace its full regularization path and recover the knots of Table 11.1 exactly on the prostate cancer data. Then recover a 160-spike signal from 1024 measurements of 4096 unknowns with FISTA and debiasing (Figure 11.13), beat plain lasso with group lasso (Figure 11.14), keep correlated features together with the elastic net, resist outliers with least absolute deviations by linear programming, Huber IRLS and Student-t EM, and finish with the Bayesian posterior, honest predictive error bars and automatic relevance determination.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_prostate

---

Built on Deep-ML.
