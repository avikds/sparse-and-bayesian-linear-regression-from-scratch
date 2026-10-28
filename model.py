"""
Sparse and Bayesian Linear Regression from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_prostate
import numpy as np

PROSTATE_NAMES = [
    "lcavol", "lweight", "age", "lbph",
    "svi", "lcp", "gleason", "pgg45"
]

PROSTATE_DATA = """
-0.579818495 2.769459 50 -1.38629436 0 -1.38629436 6 0 -0.4307829 T
-0.994252273 3.319626 58 -1.38629436 0 -1.38629436 6 0 -0.1625189 T
-0.510825624 2.691243 74 -1.38629436 0 -1.38629436 7 20 -0.1625189 T
-1.203972804 3.282789 58 -1.38629436 0 -1.38629436 6 0 -0.1625189 T
0.751416089 3.432373 62 -1.38629436 0 -1.38629436 6 0 0.3715636 T
-1.049822124 3.228826 50 -1.38629436 0 -1.38629436 6 0 0.7654678 T
0.737164066 3.473518 64 0.61518564 0 -1.38629436 6 0 0.7654678 F
0.693147181 3.539509 58 1.53686722 0 -1.38629436 6 0 0.8544153 T
-0.776528789 3.539509 47 -1.38629436 0 -1.38629436 6 0 1.0473190 F
0.223143551 3.244544 63 -1.38629436 0 -1.38629436 6 0 1.0473190 F
0.254642218 3.604138 65 -1.38629436 0 -1.38629436 6 0 1.2669476 T
-1.347073648 3.598681 63 1.26694760 0 -1.38629436 6 0 1.2669476 T
1.613429934 3.022861 63 -1.38629436 0 -0.59783700 7 30 1.2669476 T
1.477048724 2.998229 67 -1.38629436 0 -1.38629436 7 5 1.3480731 T
1.205970807 3.442019 57 -1.38629436 0 -0.43078292 7 5 1.3987169 F
1.541159072 3.061052 66 -1.38629436 0 -1.38629436 6 0 1.4469190 T
-0.415515444 3.516013 70 1.24415459 0 -0.59783700 7 30 1.4701758 T
2.288486169 3.649359 66 -1.38629436 0 0.37156356 6 0 1.4929041 T
-0.562118918 3.267666 41 -1.38629436 0 -1.38629436 6 0 1.5581446 T
0.182321557 3.825375 70 1.65822808 0 -1.38629436 6 0 1.5993876 T
1.147402453 3.419365 59 -1.38629436 0 -1.38629436 6 0 1.6389967 T
2.059238834 3.501043 60 1.47476301 0 1.34807315 7 20 1.6582281 F
-0.544727175 3.375880 59 -0.79850770 0 -1.38629436 6 0 1.6956156 T
1.781709133 3.451574 63 0.43825493 0 1.17865500 7 60 1.7137979 T
0.385262401 3.667400 69 1.59938758 0 -1.38629436 6 0 1.7316555 F
1.446918983 3.124565 68 0.30010459 0 -1.38629436 6 0 1.7664417 F
0.512823626 3.719651 65 -1.38629436 0 -0.79850770 7 70 1.8000583 T
-0.400477567 3.865979 67 1.81645208 0 -1.38629436 7 20 1.8164521 F
1.040276712 3.128951 67 0.22314355 0 0.04879016 7 80 1.8484548 T
2.409644165 3.375880 65 -1.38629436 0 1.61938824 6 0 1.8946169 T
0.285178942 4.090169 65 1.96290773 0 -0.79850770 6 0 1.9242487 T
0.182321557 3.804438 65 1.70474809 0 -1.38629436 6 0 2.0082140 F
1.275362800 3.037354 71 1.26694760 0 -1.38629436 6 0 2.0082140 T
0.009950331 3.267666 54 -1.38629436 0 -1.38629436 6 0 2.0215476 F
-0.010050336 3.216874 63 -1.38629436 0 -0.79850770 6 0 2.0476928 T
1.308332820 4.119850 64 2.17133681 0 -1.38629436 7 5 2.0856721 F
1.423108334 3.657131 73 -0.57981850 0 1.65822808 8 15 2.1575593 T
0.457424847 2.374906 64 -1.38629436 0 -1.38629436 7 15 2.1916535 T
2.660958594 4.085136 68 1.37371558 1 1.83258146 7 35 2.2137539 T
0.797507196 3.013081 56 0.93609336 0 -0.16251893 7 5 2.2772673 T
0.620576488 3.141995 60 -1.38629436 0 -1.38629436 9 80 2.2975726 T
1.442201993 3.682610 68 -1.38629436 0 -1.38629436 7 10 2.3075726 F
0.582215620 3.865979 62 1.71379793 0 -0.43078292 6 0 2.3272777 T
1.771556762 3.896909 61 -1.38629436 0 0.81093022 7 6 2.3749058 F
1.486139696 3.409496 66 1.74919985 0 -0.43078292 7 20 2.5217206 T
1.663926098 3.392829 61 0.61518564 0 -1.38629436 7 15 2.5533438 T
2.727852828 3.995445 79 1.87946505 1 2.65675691 9 100 2.5687881 T
1.163150810 4.035125 68 1.71379793 0 -0.43078292 7 40 2.5687881 F
1.745715531 3.498022 43 -1.38629436 0 -1.38629436 6 0 2.5915164 F
1.220829921 3.568123 70 1.37371558 0 -0.79850770 6 0 2.5915164 F
1.091923301 3.993603 68 -1.38629436 0 -1.38629436 7 50 2.6567569 T
1.660131027 4.234831 64 2.07317193 0 -1.38629436 6 0 2.6775910 T
0.512823626 3.633631 64 1.49290410 0 0.04879016 7 70 2.6844403 F
2.127040520 4.121473 68 1.76644166 0 1.44691898 7 40 2.6912431 F
3.153590358 3.516013 59 -1.38629436 0 -1.38629436 7 5 2.7047113 F
1.266947603 4.280132 66 2.12226154 0 -1.38629436 7 15 2.7180005 T
0.974559640 2.865054 47 -1.38629436 0 0.50077529 7 4 2.7880929 F
0.463734016 3.764682 49 1.42310833 0 -1.38629436 6 0 2.7942279 T
0.542324291 4.178226 70 0.43825493 0 -1.38629436 7 20 2.8063861 T
1.061256502 3.851211 61 1.29472717 0 -1.38629436 7 40 2.8124102 T
0.457424847 4.524502 73 2.32630162 0 -1.38629436 6 0 2.8419982 T
1.997417706 3.719651 63 1.61938824 1 1.90954250 7 40 2.8535925 F
2.775708850 3.524889 72 -1.38629436 0 1.55814462 9 95 2.8535925 T
2.034705648 3.917011 66 2.00821403 1 2.11021320 7 60 2.8820035 F
2.073171929 3.623007 64 -1.38629436 0 -1.38629436 6 0 2.8820035 F
1.458615023 3.836221 61 1.32175584 0 -0.43078292 7 20 2.8875901 F
2.022871190 3.878466 68 1.78339122 0 1.32175584 7 70 2.9204698 T
2.198335072 4.050915 72 2.30757263 0 -0.43078292 7 10 2.9626924 T
-0.446287103 4.408547 69 -1.38629436 0 -1.38629436 6 0 2.9626924 T
1.193922468 4.780383 72 2.32630162 0 -0.79850770 7 5 2.9729753 T
1.864080131 3.593194 60 -1.38629436 1 1.32175584 7 60 3.0130809 T
1.160020917 3.341093 77 1.74919985 0 -1.38629436 7 25 3.0373539 T
1.214912744 3.825375 69 -1.38629436 1 0.22314355 7 20 3.0563569 F
1.838961071 3.236716 60 0.43825493 1 1.17865500 9 90 3.0750055 F
2.999226163 3.849083 69 -1.38629436 1 1.90954250 7 20 3.2752562 T
3.141130476 3.263849 68 -0.05129329 1 2.42036813 7 50 3.3375474 T
2.010894999 4.433789 72 2.12226154 0 0.50077529 7 60 3.3928291 T
2.537657215 4.354784 78 2.32630162 0 -1.38629436 7 10 3.4355988 T
2.648300197 3.582129 69 -1.38629436 1 2.58399755 7 70 3.4578927 T
2.779440197 3.823192 63 -1.38629436 0 0.37156356 7 50 3.5130369 F
1.467874348 3.070376 66 0.55961579 0 0.22314355 7 40 3.5160131 T
2.513656063 3.473518 57 0.43825493 0 2.32727771 7 60 3.5307626 T
2.613006652 3.888754 77 -0.52763274 1 0.55961579 7 30 3.5652984 T
2.677590994 3.838376 65 1.11514159 0 1.74919985 9 70 3.5709402 F
1.562346305 3.709907 60 1.69561561 0 0.81093022 7 30 3.5876769 T
3.302849259 3.518980 64 -1.38629436 1 2.32727771 7 60 3.6309855 T
2.024193067 3.731699 58 1.63899671 0 -1.38629436 6 0 3.6800909 T
1.731655545 3.369018 62 -1.38629436 1 0.30010459 7 30 3.7123518 T
2.807593831 4.718052 65 -1.38629436 1 2.46385324 7 60 3.9843437 T
1.562346305 3.695110 76 0.93609336 1 0.81093022 7 75 3.9936030 T
3.246490992 4.101817 68 -1.38629436 0 -1.38629436 6 0 4.0298060 T
2.532902848 3.677566 61 1.34807315 1 -1.38629436 7 15 4.1295508 T
2.830267834 3.876396 68 -1.38629436 1 1.32175584 7 60 4.3851468 T
3.821003607 3.896909 44 -1.38629436 1 2.16905370 7 40 4.6844434 T
2.907447359 3.396185 52 -1.38629436 1 2.46385324 7 10 5.1431245 F
2.882563575 3.773910 68 1.55814462 1 1.55814462 7 80 5.4775090 T
3.471966453 3.974998 68 0.43825493 1 2.90416508 7 20 5.5829322 F
"""


def load_prostate():
    """Load and standardize the prostate cancer dataset."""

    rows = []

    # Parse the 8 feature columns, response (lpsa), and split label.
    for line in PROSTATE_DATA.strip().splitlines():
        fields = line.split()

        if len(fields) != 10:
            raise ValueError(f"Expected 10 fields per row, got {len(fields)}")

        features = [float(value) for value in fields[:8]]
        lpsa = float(fields[8])
        split = fields[9]

        if split not in ("T", "F"):
            raise ValueError(f"Invalid split label: {split}")

        rows.append((features, lpsa, split))

    # Select rows in their original file order without shuffling.
    train_rows = [row for row in rows if row[2] == "T"]
    test_rows = [row for row in rows if row[2] == "F"]

    X_train_raw = np.asarray([row[0] for row in train_rows], dtype=float)
    X_test_raw = np.asarray([row[0] for row in test_rows], dtype=float)

    y_train_raw = np.asarray([row[1] for row in train_rows], dtype=float)
    y_test_raw = np.asarray([row[1] for row in test_rows], dtype=float)

    # Standardize using the training features' mean and sample standard deviation.
    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0, ddof=1)

    # Avoid division by zero if a feature is constant.
    X_std = np.where(X_std == 0, 1.0, X_std)

    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # Center responses using the training response mean.
    y_mean = float(y_train_raw.mean())
    y_train = y_train_raw - y_mean
    y_test = y_test_raw - y_mean

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "names": list(PROSTATE_NAMES),
        "y_mean": y_mean,
    }

# Step 2 - ols_normal
def ols_normal(X, y):
    """
    Solve ordinary least squares using the normal equations.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Design matrix.
    y : array-like, shape (N,)
        Target vector.

    Returns
    -------
    w : ndarray, shape (D,)
        Least-squares weight vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Solve (X.T @ X) @ w = X.T @ y without forming an inverse.
    A = X.T @ X
    b = X.T @ y

    return np.linalg.solve(A, b)

# Step 3 - back_substitute
def back_substitute(R, b):
    """Solve the upper-triangular system Rw = b by back-substitution."""
    R = np.asarray(R, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]
    w = np.zeros(n, dtype=float)

    # Solve from the last row upward.
    for i in range(n - 1, -1, -1):
        w[i] = (b[i] - np.dot(R[i, i + 1:], w[i + 1:])) / R[i, i]

    return w

# Step 4 - ols_qr
def ols_qr(X, y):
    """Solve least squares using the reduced QR decomposition."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Factor X = QR, where Q has orthonormal columns and R is upper triangular.
    Q, R = np.linalg.qr(X, mode="reduced")

    # Solve Rw = Q.T @ y using the back-substitution function from Step 3.
    return back_substitute(R, Q.T @ y)

# Step 5 - ols_svd
def ols_svd(X, y, rcond=1e-12):
    """Compute the minimum-norm least-squares solution using the thin SVD."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Factor X = U @ diag(s) @ Vt.
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Retain singular values above the relative cutoff.
    if s.size == 0:
        return np.zeros(X.shape[1], dtype=float)

    keep = s > rcond * s[0]

    # Compute the minimum-norm solution using only retained singular values.
    return Vt[keep, :].T @ ((U[:, keep].T @ y) / s[keep])

# Step 6 - conditioning_report
def conditioning_report(degrees, n_points=50):
    """Compare least-squares solvers as the polynomial design becomes ill-conditioned."""
    results = []
    x = np.linspace(0.0, 1.0, n_points)

    for p in degrees:
        # Build the Vandermonde matrix with columns 1, x, ..., x**p.
        X = np.vander(x, N=p + 1, increasing=True)

        # Use known weights and generate a noise-free response.
        w_true = np.ones(p + 1, dtype=float)
        y = X @ w_true

        # Measure the condition number of the design matrix.
        log10_cond = float(np.log10(np.linalg.cond(X)))

        # Fit using each least-squares solver.
        w_normal = ols_normal(X, y)
        w_qr = ols_qr(X, y)
        w_svd = ols_svd(X, y)

        # Relative coefficient errors.
        true_norm = np.linalg.norm(w_true)
        normal_error = float(np.linalg.norm(w_normal - w_true) / true_norm)
        qr_error = float(np.linalg.norm(w_qr - w_true) / true_norm)
        svd_error = float(np.linalg.norm(w_svd - w_true) / true_norm)

        results.append({
            "degree": int(p),
            "log10_cond": log10_cond,
            "normal": normal_error,
            "qr": qr_error,
            "svd": svd_error,
        })

    return results

# Step 7 - ridge_augmented
def ridge_augmented(X, y, lam):
    """Fit ridge regression using least squares on augmented data."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N, D = X.shape

    # Append virtual observations to penalize the squared weight norm.
    X_aug = np.vstack((X, np.sqrt(lam) * np.eye(D)))
    y_aug = np.concatenate((y, np.zeros(D)))

    # Solve the augmented least-squares problem using the existing QR solver.
    return ols_qr(X_aug, y_aug)

# Step 8 - ridge_svd
def ridge_svd(X, y, lams):
    """Compute ridge solutions and effective degrees of freedom using one thin SVD."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lams = np.atleast_1d(np.asarray(lams, dtype=float))

    # Compute the thin SVD once and reuse it for every penalty.
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uy = U.T @ y
    s_squared = s ** 2

    W = np.empty((len(lams), X.shape[1]), dtype=float)
    dof = np.empty(len(lams), dtype=float)

    for k, lam in enumerate(lams):
        # Apply ridge shrinkage to each singular-value component.
        factors = s / (s_squared + lam)
        W[k] = Vt.T @ (factors * Uy)

        # Effective degrees of freedom for this penalty.
        dof[k] = np.sum(s_squared / (s_squared + lam))

    return W, dof

# Step 9 - ridge_evidence
def ridge_evidence(X, y, n_iter=500, tol=1e-10):
    """Choose the ridge penalty by maximizing the marginal likelihood."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # Initialize prior precision and noise precision.
    alpha = 1.0
    beta = 1.0 / np.var(y)

    eps = np.finfo(float).eps

    # MacKay fixed-point updates.
    for _ in range(n_iter):
        lam = alpha / beta

        # Obtain the ridge solution and effective degrees of freedom.
        W, dofs = ridge_svd(X, y, lam)
        m = W[0]
        gamma = float(dofs[0])

        # Update the prior and noise precisions.
        alpha_new = gamma / max(float(m @ m), eps)
        residual = y - X @ m
        beta_new = (N - gamma) / max(float(residual @ residual), eps)

        # Stop when both relative changes are sufficiently small.
        alpha_change = abs(alpha_new - alpha) / max(abs(alpha), eps)
        beta_change = abs(beta_new - beta) / max(abs(beta), eps)

        alpha, beta = alpha_new, beta_new

        if alpha_change <= tol and beta_change <= tol:
            break

    # Recompute the solution and degrees of freedom at final precisions.
    lam = alpha / beta
    W, dofs = ridge_svd(X, y, lam)
    m = W[0]
    gamma = float(dofs[0])

    # Calculate the residual and regularization contributions to E(m).
    residual = y - X @ m
    E_m = 0.5 * beta * float(residual @ residual)
    E_m += 0.5 * alpha * float(m @ m)

    # Eigenvalues of X.T @ X are the squared singular values of X.
    singular_values = np.linalg.svd(X, compute_uv=False)
    eigenvalues = singular_values ** 2

    # Pad with zero eigenvalues if the number of features exceeds N.
    if D > len(eigenvalues):
        eigenvalues = np.pad(eigenvalues, (0, D - len(eigenvalues)))

    log_evidence = (
        0.5 * D * np.log(alpha)
        + 0.5 * N * np.log(beta)
        - E_m
        - 0.5 * np.sum(np.log(alpha + beta * eigenvalues))
        - 0.5 * N * np.log(2.0 * np.pi)
    )

    return {
        "lam": float(lam),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "log_evidence": float(log_evidence),
    }

# Step 10 - cv_ridge
def cv_ridge(X, y, lams, n_folds=5):
    """Estimate ridge prediction error using K-fold cross-validation."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    lams = np.atleast_1d(np.asarray(lams, dtype=float))

    N = X.shape[0]
    squared_errors = np.zeros(len(lams), dtype=float)

    # Assign rows deterministically: row i belongs to fold i % n_folds.
    fold_ids = np.arange(N) % n_folds

    for k in range(n_folds):
        test_mask = fold_ids == k
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Fit all penalty values on the training folds using one SVD.
        W, _ = ridge_svd(X_train, y_train, lams)

        # Accumulate the squared prediction errors for each penalty.
        predictions = X_test @ W.T
        errors = predictions - y_test[:, None]
        squared_errors += np.sum(errors ** 2, axis=0)

    # Return mean squared prediction error over all N observations.
    return squared_errors / N

# Step 11 - soft_threshold
def soft_threshold(x, delta):
    """Apply the soft-thresholding operator elementwise."""
    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0.0)

# Step 12 - lasso_cd
def lasso_cd(X, y, lam, w0=None, max_sweeps=1000, tol=1e-8):
    """Minimize the lasso objective using cyclic coordinate descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    # Initialize weights without modifying the caller's array.
    if w0 is None:
        w = np.zeros(D, dtype=float)
    else:
        w = np.asarray(w0, dtype=float).copy()

    # Maintain the residual r = y - X @ w incrementally.
    residual = y - X @ w
    col_norms_sq = np.sum(X * X, axis=0)

    for sweep in range(1, max_sweeps + 1):
        max_change = 0.0

        for d in range(D):
            a_d = col_norms_sq[d]

            # Skip zero columns, leaving their weights unchanged.
            if a_d == 0.0:
                continue

            # Restore the contribution of the current coordinate.
            residual += X[:, d] * w[d]

            # Correlation with the residual excluding feature d.
            c_d = float(X[:, d] @ residual)

            # Update the coordinate using soft thresholding.
            new_w_d = float(soft_threshold(c_d, lam) / a_d)

            # Update the residual using the change in this coordinate.
            residual -= X[:, d] * new_w_d

            max_change = max(max_change, abs(new_w_d - w[d]))
            w[d] = new_w_d

        # Stop after the first complete sweep with a sufficiently small change.
        if max_change < tol:
            return w, sweep

    return w, max_sweeps

# Step 13 - lasso_path
def lasso_path(X, y, n_lambdas=50, eps=1e-3, tol=1e-8):
    """Compute the lasso regularization path using warm-started coordinate descent."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Maximum penalty for which the all-zero solution is optimal.
    lam_max = float(np.max(np.abs(X.T @ y)))

    # Generate penalties from lam_max down to eps * lam_max.
    lams = lam_max * np.logspace(0, np.log10(eps), n_lambdas)

    # Fit each penalty, initializing from the previous solution.
    W = np.zeros((n_lambdas, X.shape[1]), dtype=float)
    w = None

    for k, lam in enumerate(lams):
        w, _ = lasso_cd(X, y, lam, w0=w, tol=tol)
        W[k] = w

    return lams, W

# Step 14 - lasso_kkt_violation
def lasso_kkt_violation(X, y, w, lam):
    """Return the maximum KKT optimality violation for the lasso solution."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    # Correlation between each feature and the current residual.
    g = X.T @ (y - X @ w)

    # Nonzero weights require g_d = lam * sign(w_d).
    active = w != 0.0
    violations = np.empty_like(w, dtype=float)
    violations[active] = np.abs(g[active] - lam * np.sign(w[active]))

    # Zero weights require |g_d| <= lam.
    violations[~active] = np.maximum(np.abs(g[~active]) - lam, 0.0)

    return float(np.max(violations)) if violations.size else 0.0

# Step 15 - lasso_knots
def lasso_knots(X, y, tol=1e-10):
    """Compute the lasso regularization path at every active-set knot."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    N, D = X.shape

    correlations = X.T @ y
    lam_max = float(np.max(np.abs(correlations))) if D else 0.0

    # At lambda_max, the lasso solution is the zero vector.
    w = np.zeros(D, dtype=float)
    lams = [lam_max]
    W = [w.copy()]

    if D == 0 or lam_max == 0.0:
        return np.asarray(lams, dtype=float), np.asarray(W, dtype=float)

    # Start with the feature having the largest absolute correlation.
    j0 = int(np.argmax(np.abs(correlations)))
    active = [j0]
    signs = [float(np.sign(correlations[j0]))]
    lam_current = lam_max

    while lam_current > tol:
        XA = X[:, active]

        # On the current segment, w_A(lambda) = u - lambda * v.
        gram = XA.T @ XA
        rhs = XA.T @ y
        sign_vec = np.asarray(signs, dtype=float)

        try:
            u = np.linalg.solve(gram, rhs)
            v = np.linalg.solve(gram, sign_vec)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            v = np.linalg.lstsq(gram, sign_vec, rcond=None)[0]

        # Collect candidate events: active weights leaving zero or
        # inactive features joining the active set.
        candidates = []

        for k, j in enumerate(active):
            if abs(v[k]) > tol:
                lam_candidate = u[k] / v[k]
                if tol < lam_candidate < lam_current - tol:
                    candidates.append((float(lam_candidate), "leave", j, 0.0))

        residual_u = y - XA @ u
        for j in range(D):
            if j in active:
                continue

            a_j = float(X[:, j] @ residual_u)
            b_j = float(X[:, j] @ (XA @ v))

            # Solve a_j + lambda*b_j = +lambda and = -lambda.
            denom_pos = 1.0 - b_j
            if abs(denom_pos) > tol:
                lam_candidate = a_j / denom_pos
                if tol < lam_candidate < lam_current - tol:
                    candidates.append((float(lam_candidate), "join", j, 1.0))

            denom_neg = 1.0 + b_j
            if abs(denom_neg) > tol:
                lam_candidate = -a_j / denom_neg
                if tol < lam_candidate < lam_current - tol:
                    candidates.append((float(lam_candidate), "join", j, -1.0))

        if not candidates:
            # The final knot is lambda = 0.
            w = np.zeros(D, dtype=float)
            w[active] = u

            # For an underdetermined or overdetermined least-squares problem,
            # use the minimum-norm least-squares solution at lambda = 0.
            if D <= N:
                w = np.linalg.lstsq(X, y, rcond=None)[0]

            if lams[-1] > 0.0:
                lams.append(0.0)
                W.append(w.copy())
            else:
                W[-1] = w.copy()
            break

        # Move to the largest candidate below the current penalty.
        lam_next = max(item[0] for item in candidates)

        w = np.zeros(D, dtype=float)
        w[active] = u - lam_next * v
        lams.append(float(lam_next))
        W.append(w.copy())

        # Process all events occurring at this knot within tolerance.
        events = [
            item for item in candidates
            if abs(item[0] - lam_next) <= tol
        ]

        # Remove variables whose coefficients reach zero.
        for _, event_type, j, _ in events:
            if event_type == "leave" and j in active:
                k = active.index(j)
                active.pop(k)
                signs.pop(k)

        # Add joining variables with the sign of their residual correlation.
        residual = y - X @ w
        for _, event_type, j, candidate_sign in events:
            if event_type == "join" and j not in active:
                corr = float(X[:, j] @ residual)
                new_sign = float(np.sign(corr))
                if new_sign == 0.0:
                    new_sign = candidate_sign
                active.append(j)
                signs.append(new_sign)

        lam_current = lam_next

    # Ensure the returned path ends at lambda = 0.
    if lams[-1] > 0.0:
        if active:
            XA = X[:, active]
            gram = XA.T @ XA
            try:
                u = np.linalg.solve(gram, XA.T @ y)
            except np.linalg.LinAlgError:
                u = np.linalg.lstsq(gram, XA.T @ y, rcond=None)[0]
            w = np.zeros(D, dtype=float)
            w[active] = u
        else:
            w = np.zeros(D, dtype=float)

        if D <= N:
            w = np.linalg.lstsq(X, y, rcond=None)[0]

        lams.append(0.0)
        W.append(w.copy())

    return np.asarray(lams, dtype=float), np.asarray(W, dtype=float)

# Step 16 - fista
def fista(X, y, prox, lam, L, n_iter=500):
    """Minimize 0.5 * ||y - Xw||^2 + penalty(w) using FISTA."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    D = X.shape[1]

    w = np.zeros(D, dtype=float)
    z = w.copy()
    t = 1.0

    for _ in range(n_iter):
        # Gradient step evaluated at the extrapolated point z.
        gradient = X.T @ (X @ z - y)
        w_new = prox(z - gradient / L, lam / L)

        # Nesterov acceleration.
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)

        w = w_new
        t = t_new

    return w

# Step 17 - block_signal
def block_signal(D, block_size, n_active, seed, kind="sign"):
    """Generate a block-sparse signal and its group assignments."""
    rng = np.random.default_rng(seed)
    n_blocks = D // block_size

    # Select active blocks without replacement and sort them.
    active = np.sort(rng.choice(n_blocks, n_active, replace=False))

    # Generate all active values at once, following the specified RNG order.
    if kind == "sign":
        values = rng.choice([-1.0, 1.0], size=n_active * block_size)
    elif kind == "gaussian":
        values = rng.standard_normal(n_active * block_size)
    elif kind == "ones":
        values = np.ones(n_active * block_size, dtype=float)
    else:
        raise ValueError("kind must be 'sign', 'gaussian', or 'ones'")

    # Place consecutive chunks of values into the active blocks.
    w = np.zeros(D, dtype=float)
    for i, block in enumerate(active):
        start = int(block) * block_size
        end = start + block_size
        value_start = i * block_size
        value_end = value_start + block_size
        w[start:end] = values[value_start:value_end]

    groups = np.arange(D) // block_size
    return w, groups

# Step 18 - sensing_problem
def sensing_problem(w_true, N, noise, seed):
    """Generate noisy measurements using an orthonormal-row sensing matrix."""
    w_true = np.asarray(w_true, dtype=float)
    D = w_true.shape[0]
    rng = np.random.default_rng(seed)

    # Generate the random design matrix.
    A = rng.standard_normal((N, D))

    # Orthonormalize the rows using Cholesky factorization and a linear solve.
    Lc = np.linalg.cholesky(A @ A.T)
    X = np.linalg.solve(Lc, A)

    # Generate noisy measurements using the same random generator.
    y = X @ w_true + noise * rng.standard_normal(N)

    return X, y

# Step 19 - debias
def debias(X, y, w_hat, thresh=1e-6):
    """Remove lasso shrinkage by refitting least squares on the selected support."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w_hat = np.asarray(w_hat, dtype=float)

    D = X.shape[1]
    w = np.zeros(D, dtype=float)

    # Select features whose estimated coefficients exceed the threshold.
    support = np.abs(w_hat) > thresh

    # Return zeros if no features are selected.
    if not np.any(support):
        return w

    # Refit least squares on the selected features.
    w[support] = ols_svd(X[:, support], y)

    return w

# Step 20 - recovery_report
def recovery_report(w_hat, w_true, thresh=1e-6):
    """Evaluate estimation error and support recovery for a sparse signal."""
    w_hat = np.asarray(w_hat, dtype=float)
    w_true = np.asarray(w_true, dtype=float)

    # Mean squared error across all coordinates.
    mse = float(np.mean((w_hat - w_true) ** 2))

    # Identify selected and truly active coordinates.
    selected_mask = np.abs(w_hat) > thresh
    active_mask = w_true != 0.0

    # Count support recovery outcomes.
    selected = int(np.count_nonzero(selected_mask))
    true_pos = int(np.count_nonzero(selected_mask & active_mask))
    false_pos = int(np.count_nonzero(selected_mask & ~active_mask))
    missed = int(np.count_nonzero(~selected_mask & active_mask))

    return {
        "mse": mse,
        "selected": selected,
        "true_pos": true_pos,
        "false_pos": false_pos,
        "missed": missed,
    }

# Step 21 - group_soft_threshold
def group_soft_threshold(v, groups, t):
    """Apply group-wise soft thresholding for the group-lasso penalty."""
    v = np.asarray(v, dtype=float)
    groups = np.asarray(groups)

    out = np.zeros_like(v, dtype=float)

    for g in np.unique(groups):
        mask = groups == g
        group_norm = np.linalg.norm(v[mask])

        if group_norm > t:
            out[mask] = (1.0 - t / group_norm) * v[mask]

    return out

# Step 22 - elastic_net
def elastic_net(X, y, lam1, lam2, tol=1e-10, max_sweeps=5000):
    """Fit elastic net by reducing it to lasso on augmented, scaled data."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    D = X.shape[1]

    c = (1.0 + lam2) ** -0.5

    # Augment the design matrix to incorporate the L2 penalty.
    X_aug = c * np.vstack((X, np.sqrt(lam2) * np.eye(D)))
    y_aug = np.concatenate((y, np.zeros(D)))

    # lasso_cd minimizes 0.5 * squared_error + lambda * L1_norm.
    w_scaled, _ = lasso_cd(
        X_aug,
        y_aug,
        lam=c * lam1 / 2.0,
        tol=tol,
        max_sweeps=max_sweeps
    )

    # Transform the coefficients back to the elastic-net solution.
    return c * w_scaled

# Step 23 - lad_lp
def lad_lp(X, y):
    """Fit least absolute deviations regression using linear programming."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    N, D = X.shape

    # Variables are v = (w_1, ..., w_D, e_1, ..., e_N).
    c = np.concatenate((np.zeros(D), np.ones(N)))

    # Construct the 2N interleaved inequality constraints.
    A = np.zeros((2 * N, D + N), dtype=float)
    b = np.empty(2 * N, dtype=float)

    for i in range(N):
        # x_i @ w - e_i <= y_i
        A[2 * i, :D] = X[i]
        A[2 * i, D + i] = -1.0
        b[2 * i] = y[i]

        # -x_i @ w - e_i <= -y_i
        A[2 * i + 1, :D] = -X[i]
        A[2 * i + 1, D + i] = -1.0
        b[2 * i + 1] = -y[i]

    # All variables are declared free; the constraints enforce e_i >= |y_i - x_i @ w|.
    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        bounds=[(None, None)] * (D + N),
        method="highs"
    )

    if not result.success:
        raise RuntimeError(f"LAD linear program failed: {result.message}")

    return result.x[:D]

# Step 24 - huber_irls
def huber_irls(X, y, delta, n_iter=100):
    """Fit Huber-loss regression using iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initialize with the ordinary least-squares solution.
    w = ols_svd(X, y)

    for _ in range(n_iter):
        # Compute absolute residual magnitudes.
        residuals = np.abs(y - X @ w)

        # Assign weight 1 inside the Huber threshold and delta / |r| outside.
        weights = np.ones_like(residuals)
        mask = residuals > delta
        weights[mask] = delta / residuals[mask]

        # Solve weighted least squares by scaling rows with sqrt(weights).
        sqrt_weights = np.sqrt(weights)
        X_weighted = X * sqrt_weights[:, None]
        y_weighted = y * sqrt_weights

        w = ols_svd(X_weighted, y_weighted)

    return w

# Step 25 - student_t_em
def student_t_em(X, y, nu, n_iter=200):
    """Fit Student-t linear regression using expectation-maximization."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    N = X.shape[0]

    # Initialize weights with ordinary least squares and estimate noise variance.
    w = ols_svd(X, y)
    residuals = y - X @ w
    sigma2 = float(np.mean(residuals ** 2))

    for _ in range(n_iter):
        # E-step: estimate the latent precision weights.
        residuals = y - X @ w
        z = (nu + 1.0) / (nu + residuals ** 2 / sigma2)

        # M-step: solve weighted least squares using scaled rows.
        sqrt_z = np.sqrt(z)
        X_weighted = X * sqrt_z[:, None]
        y_weighted = y * sqrt_z
        w = ols_svd(X_weighted, y_weighted)

        # Update the noise variance using the new regression weights.
        residuals = y - X @ w
        sigma2 = float(np.mean(z * residuals ** 2))

    return w, sigma2

