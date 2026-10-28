"""
Sparse and Bayesian Linear Regression from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sparse and Bayesian Linear Regression from Scratch (Murphy, Probabilistic Machine Learning, chapter 11).
Runs every experiment of the project with your step functions and prints the headline numbers."""
import numpy as np


def test_mse(w, X, y):
    return float(np.mean((X @ w - y) ** 2))


def main():
    np.set_printoptions(suppress=True, linewidth=150)

    # --- Part 1: least squares three ways ---
    P = load_prostate()
    Xtr, ytr, Xte, yte = P["X_train"], P["y_train"], P["X_test"], P["y_test"]
    names = P["names"]
    w_ols = ols_svd(Xtr, ytr)
    agree = max(np.abs(ols_normal(Xtr, ytr) - w_ols).max(), np.abs(ols_qr(Xtr, ytr) - w_ols).max())
    print(f"Prostate: {len(ytr)} training and {len(yte)} test men, {Xtr.shape[1]} standardized features")
    print("OLS weights (last row of Table 11.1):", np.round(w_ols, 4).tolist(), f"| three solvers agree to {agree:.0e}")
    print("Conditioning stress test (relative error of the weights, noise-free polynomial fits):")
    for r in conditioning_report([4, 8, 10, 12]):
        print(f"  degree {r['degree']:2d}  log10 cond {r['log10_cond']:5.2f}  normal eq {r['normal']:.1e}  QR {r['qr']:.1e}  SVD {r['svd']:.1e}")

    # --- Part 2: ridge, its degrees of freedom, evidence vs cross-validation ---
    lams = np.logspace(-2, 3, 61)
    cv = cv_ridge(Xtr, ytr, lams)
    lam_cv = float(lams[np.argmin(cv)])
    ev = ridge_evidence(Xtr, ytr)
    W, dof = ridge_svd(Xtr, ytr, [lam_cv, ev["lam"]])
    print(f"Ridge by 5-fold CV:   lambda={lam_cv:.3f}  dof={dof[0]:.2f}  test MSE={test_mse(W[0], Xte, yte):.4f}")
    print(f"Ridge by evidence:    lambda={ev['lam']:.3f}  dof={dof[1]:.2f}  noise sd={ev['beta'] ** -0.5:.3f}  "
          f"log evidence={ev['log_evidence']:.2f}  test MSE={test_mse(W[1], Xte, yte):.4f}")
    print(f"OLS:                  test MSE={test_mse(w_ols, Xte, yte):.4f}")

    # --- Part 3: the lasso path, exactly ---
    knots, Wk = lasso_knots(Xtr, ytr)
    print("Lasso knots (Table 11.1): lambda and coefficients where the active set changes")
    for lam, row in zip(knots, Wk):
        print(f"  {lam:8.4f}  " + " ".join(f"{v:7.4f}" for v in row + 0.0))
    entered = []
    for row in Wk[1:]:
        entered += [names[j] for j in np.flatnonzero(row) if names[j] not in entered]
    print("  order of entry:", ", ".join(entered))
    gap = max(lasso_kkt_violation(Xtr, ytr, lasso_cd(Xtr, ytr, lam, tol=1e-12)[0], lam) for lam in knots[1:-1])
    print(f"  coordinate descent at every knot satisfies the KKT conditions to {gap:.0e}")
    lam_grid = float(np.max(np.abs(Xtr.T @ ytr))) * np.logspace(0, -3, 40)
    fold = np.arange(len(ytr)) % 5
    cv_l = np.zeros(len(lam_grid))
    for k in range(5):
        tr, va = fold != k, fold == k
        w = np.zeros(Xtr.shape[1])
        for i, lam in enumerate(lam_grid):
            w, _ = lasso_cd(Xtr[tr], ytr[tr], lam, w0=w)
            cv_l[i] += np.sum((Xtr[va] @ w - ytr[va]) ** 2)
    lam_l = float(lam_grid[np.argmin(cv_l)])
    w_l, _ = lasso_cd(Xtr, ytr, lam_l)
    print(f"Lasso by 5-fold CV: lambda={lam_l:.3f}  dropped {[names[j] for j in np.flatnonzero(w_l == 0)]}  "
          f"test MSE={test_mse(w_l, Xte, yte):.4f}")

    # --- Part 4: compressed sensing (Figure 11.13) ---
    w_true, _ = block_signal(4096, 1, 160, seed=1)
    X, y = sensing_problem(w_true, 1024, 0.01, seed=101)
    lam_max = float(np.max(np.abs(X.T @ y)))
    w_l1 = fista(X, y, soft_threshold, 0.1 * lam_max, 1.0, 500)
    r_mn = recovery_report(X.T @ y, w_true)
    r_l1 = recovery_report(w_l1, w_true)
    r_db = recovery_report(debias(X, y, w_l1), w_true)
    print("Figure 11.13: D=4096 unknowns, N=1024 measurements, 160 spikes, noise sd 0.01")
    print(f"  minimum-norm MSE={r_mn['mse']:.4f} | lasso (lambda=0.1 lambda_max) MSE={r_l1['mse']:.4f}, "
          f"{r_l1['selected']} selected, {r_l1['true_pos']} true, {r_l1['missed']} missed | debiased MSE={r_db['mse']:.1e}")
    print("  the prediction-vs-selection trade-off:")
    for frac in (0.02, 0.05, 0.1, 0.2, 0.4):
        wf = fista(X, y, soft_threshold, frac * lam_max, 1.0, 500)
        rf, rd = recovery_report(wf, w_true), recovery_report(debias(X, y, wf), w_true)
        print(f"    lambda={frac:4.2f} lambda_max  lasso MSE={rf['mse']:.1e}  selected={rf['selected']:3d}  "
              f"false pos={rf['false_pos']:3d}  missed={rf['missed']:3d}  debiased MSE={rd['mse']:.1e}")

    # --- Part 5: group structure (Figures 11.14 and 11.15) and correlated features ---
    for kind, fig in (("gaussian", "11.14"), ("ones", "11.15")):
        w_true, groups = block_signal(4096, 64, 8, seed=7, kind=kind)
        X, y = sensing_problem(w_true, 1024, 0.01, seed=8)
        w_l1 = fista(X, y, soft_threshold, 0.427, 1.0, 500)
        w_gl = fista(X, y, lambda v, t: group_soft_threshold(v, groups, t), 0.427, 1.0, 500)
        r1, rg = recovery_report(debias(X, y, w_l1), w_true), recovery_report(debias(X, y, w_gl), w_true)
        print(f"Figure {fig} ({kind} blocks, 8 of 64 groups active): debiased lasso MSE={r1['mse']:.4f} "
              f"({r1['selected']} coords) | debiased group lasso MSE={rg['mse']:.1e} ({len(set(groups[w_gl != 0]))} groups)")
    rng = np.random.default_rng(3)
    z = rng.standard_normal(100)
    Xc = np.column_stack([z + 0.01 * rng.standard_normal(100), z + 0.01 * rng.standard_normal(100), rng.standard_normal((100, 3))])
    yc = 2 * z + 0.5 * Xc[:, 2] + 0.1 * rng.standard_normal(100)
    print("Two near-identical features: lasso", (np.round(lasso_cd(Xc, yc, 5.0, tol=1e-12)[0], 3) + 0.0).tolist(),
          "| elastic net", (np.round(elastic_net(Xc, yc, 10.0, 5.0), 3) + 0.0).tolist())

    # --- Part 6: robust regression (Figure 11.19) ---
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 1, 40))
    y = 1 + 2 * x + 0.1 * rng.standard_normal(40)
    idx = np.where(x > 0.75)[0][:6]
    y[idx] -= 4.0
    X = np.column_stack([np.ones(40), x])
    w_t, s2_t = student_t_em(X, y, 3.0)
    print("Line y = 1 + 2x with 6 outliers at the bottom (true slope 2):")
    for name, w in (("least squares", ols_svd(X, y)), ("LAD by linear programming", lad_lp(X, y)),
                    ("Huber IRLS (delta 0.1)", huber_irls(X, y, 0.1)), ("Student-t EM (nu 3)", w_t)):
        print(f"  {name:27s} intercept={w[0]:6.3f}  slope={w[1]:6.3f}")
    print(f"  Student-t noise scale sigma={np.sqrt(s2_t):.3f} (clean noise sd 0.1)")

    # --- Part 7: Bayesian linear regression and ARD ---
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1, 1, 100)
    ys = -0.3 + 0.5 * xs + 0.2 * rng.standard_normal(100)
    means, covs = blr_sequential(np.column_stack([np.ones(100), xs]), ys, np.zeros(2), np.eye(2) / 2.0, 0.04)
    print("Figure 11.20, sequential posterior over (w0, w1), truth (-0.3, 0.5):")
    for n in (1, 2, 10, 100):
        S = covs[n - 1]
        print(f"  after {n:3d} points: mean={np.round(means[n - 1], 3).tolist()}  sd={np.round(np.sqrt(np.diag(S)), 3).tolist()}  "
              f"corr={S[0, 1] / np.sqrt(S[0, 0] * S[1, 1]):+.3f}")
    f = lambda t: 1.0 + 2.0 * t - 0.5 * t ** 2
    poly = lambda t: np.column_stack([np.ones_like(t), t, t ** 2])
    scores = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        xt = rng.uniform(-2, 2, 10)
        yt = f(xt) + rng.standard_normal(10)
        xi = rng.uniform(-2, 2, 500)
        xo = np.concatenate([rng.uniform(-6, -3, 250), rng.uniform(3, 6, 250)])
        ri = predictive_nll_report(poly(xt), yt, poly(xi), f(xi) + rng.standard_normal(500), 1.0, 100.0)
        ro = predictive_nll_report(poly(xt), yt, poly(xo), f(xo) + rng.standard_normal(500), 1.0, 100.0)
        scores.append([ri["plugin"], ri["bayes"], ro["plugin"], ro["bayes"]])
    s = np.mean(scores, axis=0)
    print("Figure 11.21, quadratic fit to 10 points on [-2, 2], mean test NLL over 20 seeds:")
    print(f"  inside the data: plug-in {s[0]:.3f} vs posterior predictive {s[1]:.3f} | "
          f"extrapolating to 3 < |x| < 6: plug-in {s[2]:.2f} vs posterior predictive {s[3]:.2f}")
    rng = np.random.default_rng(5)
    Xa = rng.standard_normal((100, 20))
    wa = np.zeros(20)
    wa[:5] = [2.0, -1.5, 1.0, 0.5, -0.8]
    ya = Xa @ wa + 0.5 * rng.standard_normal(100)
    w_ard, alpha, hist = ard(Xa, ya, 0.25)
    Xn = rng.standard_normal((2000, 20))
    print(f"ARD, 20 features of which 5 matter: kept {int((alpha < 1e8).sum())}, log evidence {hist[0]:.1f} -> {hist[-1]:.1f}, "
          f"test MSE {np.mean((Xn @ (w_ard - wa)) ** 2):.4f} vs least squares {np.mean((Xn @ (ols_svd(Xa, ya) - wa)) ** 2):.4f}")
    s2 = float(np.sum((ytr - Xtr @ w_ols) ** 2) / (len(ytr) - Xtr.shape[1]))
    w_p, alpha_p, _ = ard(Xtr, ytr, s2)
    print(f"ARD on prostate (sigma^2={s2:.3f} from OLS): pruned {[names[j] for j in range(8) if alpha_p[j] >= 1e8]}, "
          f"test MSE={test_mse(w_p, Xte, yte):.4f}")


if __name__ == "__main__":
    main()

