# Sparse and Bayesian Linear Regression from Scratch

Rebuild the linear-regression toolkit of Murphy's Probabilistic Machine Learning (chapter 11) in NumPy and reproduce its results. Solve least squares by normal equations, QR back-substitution and the SVD, and break the normal equations on an ill-conditioned design. Fit ridge regression through virtual data and the SVD, measure its effective degrees of freedom, and pick its strength by the evidence and by cross-validation. Write the lasso shooting algorithm, trace its full regularization path and recover the knots of Table 11.1 exactly on the prostate cancer data. Then recover a 160-spike signal from 1024 measurements of 4096 unknowns with FISTA and debiasing (Figure 11.13), beat plain lasso with group lasso (Figure 11.14), keep correlated features together with the elastic net, resist outliers with least absolute deviations by linear programming, Huber IRLS and Student-t EM, and finish with the Bayesian posterior, honest predictive error bars and automatic relevance determination.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_prostate
- [x] **2.** ols_normal
- [x] **3.** back_substitute
- [x] **4.** ols_qr
- [x] **5.** ols_svd
- [x] **6.** conditioning_report
- [x] **7.** ridge_augmented
- [x] **8.** ridge_svd
- [x] **9.** ridge_evidence
- [x] **10.** cv_ridge
- [x] **11.** soft_threshold
- [x] **12.** lasso_cd
- [x] **13.** lasso_path
- [x] **14.** lasso_kkt_violation
- [x] **15.** lasso_knots
- [x] **16.** fista
- [x] **17.** block_signal
- [x] **18.** sensing_problem
- [x] **19.** debias
- [x] **20.** recovery_report
- [x] **21.** group_soft_threshold
- [x] **22.** elastic_net
- [x] **23.** lad_lp
- [x] **24.** huber_irls
- [x] **25.** student_t_em
- [x] **26.** blr_posterior
- [x] **27.** blr_sequential
- [x] **28.** blr_predictive
- [x] **29.** predictive_nll_report
- [x] **30.** ard

## Results

```
Prostate: 67 training and 30 test men, 8 standardized features
OLS weights (last row of Table 11.1): [0.7164, 0.2926, -0.1425, 0.212, 0.3096, -0.289, -0.0209, 0.2773] | three solvers agree to 2e-15
Conditioning stress test (relative error of the weights, noise-free polynomial fits):
  degree  4  log10 cond  2.81  normal eq 1.0e-11  QR 2.6e-14  SVD 3.3e-14
  degree  8  log10 cond  5.80  normal eq 3.7e-06  QR 1.8e-11  SVD 2.3e-11
  degree 10  log10 cond  7.31  normal eq 1.0e-03  QR 5.3e-10  SVD 4.4e-10
  degree 12  log10 cond  8.83  normal eq 1.1e+01  QR 8.4e-09  SVD 8.3e-09
Ridge by 5-fold CV:   lambda=3.162  dof=7.27  test MSE=0.5000
Ridge by evidence:    lambda=5.337  dof=6.87  noise sd=0.707  log evidence=-80.78  test MSE=0.4932
OLS:                  test MSE=0.5213
Lasso knots (Table 11.1): lambda and coefficients where the active set changes
   58.4439   0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
   30.1993   0.4279  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
   23.8878   0.5015  0.0735  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
   14.0587   0.5610  0.1878  0.0000  0.0000  0.0930  0.0000  0.0000  0.0000
   13.8132   0.5622  0.1890  0.0000  0.0036  0.0963  0.0000  0.0000  0.0000
    4.0077   0.5797  0.2456  0.0000  0.1435  0.2003  0.0000  0.0000  0.0901
    3.0154   0.5864  0.2572 -0.0321  0.1639  0.2082  0.0000  0.0000  0.1066
    0.3278   0.6994  0.2910 -0.1337  0.2062  0.3003 -0.2565  0.0000  0.2452
    0.0000   0.7164  0.2926 -0.1425  0.2120  0.3096 -0.2890 -0.0209  0.2773
  order of entry: lcavol, lweight, svi, lbph, pgg45, age, lcp, gleason
  coordinate descent at every knot satisfies the KKT conditions to 3e-11
Lasso by 5-fold CV: lambda=0.410  dropped ['gleason']  test MSE=0.5052
Figure 11.13: D=4096 unknowns, N=1024 measurements, 160 spikes, noise sd 0.01
  minimum-norm MSE=0.0296 | lasso (lambda=0.1 lambda_max) MSE=0.0027, 209 selected, 160 true, 0 missed | debiased MSE=3.3e-05
  the prediction-vs-selection trade-off:
    lambda=0.02 lambda_max  lasso MSE=2.3e-04  selected=427  false pos=267  missed=  0  debiased MSE=1.5e-04
    lambda=0.05 lambda_max  lasso MSE=7.4e-04  selected=236  false pos= 76  missed=  0  debiased MSE=5.5e-05
    lambda=0.10 lambda_max  lasso MSE=2.7e-03  selected=209  false pos= 49  missed=  0  debiased MSE=3.3e-05
    lambda=0.20 lambda_max  lasso MSE=1.1e-02  selected=199  false pos= 39  missed=  0  debiased MSE=2.7e-05
    lambda=0.40 lambda_max  lasso MSE=3.0e-02  selected=109  false pos= 14  missed= 65  debiased MSE=1.7e-02
Figure 11.14 (gaussian blocks, 8 of 64 groups active): debiased lasso MSE=0.0778 (64 coords) | debiased group lasso MSE=8.6e-05 (8 groups)
Figure 11.15 (ones blocks, 8 of 64 groups active): debiased lasso MSE=0.1272 (70 coords) | debiased group lasso MSE=8.6e-05 (8 groups)
Two near-identical features: lasso [1.962, 0.0, 0.424, 0.0, 0.0] | elastic net [0.964, 0.956, 0.406, 0.0, 0.0]
Line y = 1 + 2x with 6 outliers at the bottom (true slope 2):
  least squares               intercept= 1.677  slope=-0.678
  LAD by linear programming   intercept= 1.037  slope= 1.924
  Huber IRLS (delta 0.1)      intercept= 1.039  slope= 1.883
  Student-t EM (nu 3)         intercept= 1.003  slope= 2.016
  Student-t noise scale sigma=0.116 (clean noise sd 0.1)
Figure 11.20, sequential posterior over (w0, w1), truth (-0.3, 0.5):
  after   1 points: mean=[-0.373, -0.102]  sd=[0.259, 0.684]  corr=-0.669
  after   2 points: mean=[-0.56, 0.41]  sd=[0.142, 0.338]  corr=+0.213
  after  10 points: mean=[-0.347, 0.405]  sd=[0.064, 0.098]  corr=-0.155
  after 100 points: mean=[-0.314, 0.486]  sd=[0.02, 0.033]  corr=-0.157
Figure 11.21, quadratic fit to 10 points on [-2, 2], mean test NLL over 20 seeds:
  inside the data: plug-in 1.638 vs posterior predictive 1.579 | extrapolating to 3 < |x| < 6: plug-in 33.33 vs posterior predictive 3.03
ARD, 20 features of which 5 matter: kept 8, log evidence -130.0 -> -90.5, test MSE 0.0380 vs least squares 0.0753
ARD on prostate (sigma^2=0.499 from OLS): pruned ['gleason'], test MSE=0.4623
```
