"""
Multinomial Logit (MNL) estimation for the Shopper experiment engine.

Utility of product j in experiment n:

    V_j = beta_0 + beta_pos * position_j + beta_price * price_j
        + beta_rating * rating_j + beta_logrev * log(1 + review_count_j) # need to try without log too!
        + beta_sp * is_sponsored_j + beta_bs * is_best_seller_j
        + beta_op * is_overall_pick_j

Two specs via outside_good argument:

  outside_good=True  — outside good included, beta_0 estimated.
      Requires ~30% no_purchase rate for beta_0 to be identified.
      P(j) = exp(V_j) / (1 + sum_k exp(V_k))
      P(no_purchase) = 1 / (1 + sum_k exp(V_k))

  outside_good=False — no outside good, beta_0 dropped.
      no_purchase experiments are excluded.
      P(j) = exp(V_j) / sum_k exp(V_k)

Observability note: continuous features are normalized (mean 0, std 1) inside
prepare_experiments, and the normalization parameters (mu, sigma per feature) are
returned alongside experiments so that V_j values from compute_utilities can be
mapped back to raw units if needed.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CONT_FEATURES = ["position", "price", "rating", "log_review_count"]
BIN_FEATURES  = ["is_sponsored", "is_best_seller", "is_overall_pick"]
FEATURES      = CONT_FEATURES + BIN_FEATURES


def prepare_experiments(df, category=None, variant="full", outside_good=True):
    sub = df[df["variant"] == variant].copy()
    if category:
        sub = sub[sub["category"] == category]

    sub = sub.dropna(subset=["price", "rating", "review_count", "position"])
    if sub.empty:
        raise ValueError("No rows remain after filtering — check category/variant.")

    sub["log_review_count"] = np.log1p(sub["review_count"].astype(float))

    # Normalize continuous features (mean 0, std 1) so coefficients are comparable.
    norm_params = {}
    for col in CONT_FEATURES:
        mu    = sub[col].mean()
        sigma = sub[col].std()
        norm_params[col] = {"mu": mu, "sigma": sigma}
        sub[col] = (sub[col] - mu) / max(sigma, 1e-10)

    for col in BIN_FEATURES:
        sub[col] = sub[col].astype(float)

    experiments = []
    for _, group in sub.groupby("experiment_id"):
        X           = group[FEATURES].values.astype(float)
        chosen_mask = group["chosen"].values

        chosen_idx = int(np.where(chosen_mask)[0][0]) if chosen_mask.any() else None

        if not outside_good and chosen_idx is None:
            continue

        experiments.append({"X": X, "chosen_idx": chosen_idx})

    return experiments, norm_params


def _nll_and_grad(betas, experiments, outside_good):
    """
    Returns (negative log-likelihood, gradient) for L-BFGS-B.

    Analytical gradient avoids finite-difference noise and ensures the optimizer
    satisfies the first-order condition at convergence:
        sum_n P_n(no_purchase) == n_no_purchase   [when outside_good=True]
    Nelder-Mead (gradient-free) frequently fails this condition on this data.
    """
    beta_0    = betas[0]  if outside_good else 0.0
    beta_rest = betas[1:] if outside_good else betas

    total_nll = 0.0
    grad = np.zeros_like(betas)

    for exp in experiments:
        X          = exp["X"]           # (J, K)
        chosen_idx = exp["chosen_idx"]

        utilities = beta_0 + X @ beta_rest   # (J,)
        exp_u     = np.exp(utilities)
        sum_exp   = np.sum(exp_u)
        denom     = (1.0 + sum_exp) if outside_good else sum_exp

        if chosen_idx is None:
            prob = 1.0 / denom
        else:
            prob = exp_u[chosen_idx] / denom

        total_nll -= np.log(prob + 1e-300)

        # Inside-good probabilities: P_n(j) = exp(V_j) / denom
        P_inside = exp_u / denom           # (J,)

        # Gradient w.r.t. beta_rest: E_n[x_k] - x_{chosen,k}
        expected_x = X.T @ P_inside        # (K,)
        chosen_x   = X[chosen_idx] if chosen_idx is not None else np.zeros(X.shape[1])
        grad_rest  = expected_x - chosen_x # (K,)

        if outside_good:
            # Gradient of NLL w.r.t. beta_0: y_n(0) - P_n(0)
            # At MLE: sum_n P_n(0) == n_no_purchase (first-order condition)
            P_0   = 1.0 / denom
            y_0   = 1.0 if chosen_idx is None else 0.0
            grad[0]  += y_0 - P_0
            grad[1:] += grad_rest
        else:
            grad += grad_rest

    return total_nll, grad


def log_likelihood(betas, experiments, outside_good=True):
    """Negative log-likelihood only (kept for compatibility)."""
    nll, _ = _nll_and_grad(betas, experiments, outside_good)
    return nll


def compute_utilities(experiments, betas, outside_good=True):
    """
    Return per-experiment V_j arrays given estimated betas.

    Values are in normalized feature space (see module docstring). Each entry
    mirrors the structure of experiments from prepare_experiments.

    Returns: list of {"utilities": np.array, "chosen_idx": int or None}
    """
    beta_0    = betas[0]  if outside_good else 0.0
    beta_rest = betas[1:] if outside_good else betas

    return [
        {"utilities": beta_0 + exp["X"] @ beta_rest, "chosen_idx": exp["chosen_idx"]}
        for exp in experiments
    ]


def predicted_no_purchase_rate(experiments, betas, outside_good=True):
    """
    Compute the model-implied mean P(no_purchase) given fitted betas.

    At the true MLE, this must equal the observed no_purchase rate (first-order
    condition). Use this as a convergence diagnostic: a large discrepancy means
    the optimizer did not find the true MLE.

    Returns: (mean implied P(no_purchase), list of per-experiment values)
    """
    if not outside_good:
        return None, []

    beta_0    = betas[0]
    beta_rest = betas[1:]

    rates = []
    for exp in experiments:
        utilities = beta_0 + exp["X"] @ beta_rest
        sum_exp   = np.sum(np.exp(utilities))
        rates.append(1.0 / (1.0 + sum_exp))

    return float(np.mean(rates)), rates


def fit_mnl(df, category=None, variant="full", outside_good=True):
    experiments, norm_params = prepare_experiments(
        df, category=category, variant=variant, outside_good=outside_good
    )

    feature_names = (["beta_0"] + FEATURES) if outside_good else FEATURES
    initial_betas = np.zeros(len(feature_names))

    result = minimize(
        _nll_and_grad,
        x0=initial_betas,
        args=(experiments, outside_good),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )

    beta_hat = result.x

    if outside_good:
        ll_null = sum(np.log(1.0 / (len(exp["X"]) + 1)) for exp in experiments)
    else:
        ll_null = sum(np.log(1.0 / len(exp["X"])) for exp in experiments)

    ll        = -result.fun
    pseudo_r2 = 1.0 - ll / ll_null

    return {
        "summary":        pd.DataFrame({"feature": feature_names, "coef": beta_hat}),
        "log_likelihood": ll,
        "ll_null":        ll_null,
        "pseudo_r2":      pseudo_r2,
        "n_experiments":  len(experiments),
        "n_no_purchase":  sum(1 for exp in experiments if exp["chosen_idx"] is None),
        "outside_good":   outside_good,
        "converged":      result.success,
        "beta":           beta_hat,
        "experiments":    experiments,
        "norm_params":    norm_params,
    }


def print_results(res, label=""):
    header = f"MNL Results — {label}" if label else "MNL Results"
    spec   = "with outside good" if res["outside_good"] else "no outside good"
    print("=" * 60)
    print(f"{header}  [{spec}]")
    print("=" * 60)
    print(f"  N experiments : {res['n_experiments']:,}")
    if res["outside_good"]:
        print(f"  No-purchase   : {res['n_no_purchase']:,}  "
              f"({100 * res['n_no_purchase'] / res['n_experiments']:.1f}%)")
    print(f"  Log-likelihood: {res['log_likelihood']:.3f}")
    print(f"  LL (null)     : {res['ll_null']:.3f}")
    print(f"  McFadden R²   : {res['pseudo_r2']:.4f}")
    print(f"  Converged     : {res['converged']}")
    print()
    print(f"  {'feature':<20}  {'coef':>9}")
    print(f"  {'-'*20}  {'-'*9}")
    for _, row in res["summary"].iterrows():
        print(f"  {row['feature']:<20}  {row['coef']:>9.4f}")
    print()


if __name__ == "__main__":
    import argparse
    from src.analysis_helper import load_results_to_dataframe

    parser = argparse.ArgumentParser()
    parser.add_argument("--category",        default=None)
    parser.add_argument("--variant",         default="full")
    parser.add_argument("--no-outside-good", action="store_true")
    parser.add_argument("--all-categories",  action="store_true")
    args = parser.parse_args()

    outside_good = not args.no_outside_good
    df = load_results_to_dataframe()

    if args.all_categories:
        for cat in sorted(df["category"].dropna().unique()):
            try:
                res = fit_mnl(df, category=cat, variant=args.variant,
                              outside_good=outside_good)
                print_results(res, label=cat)
            except Exception as e:
                print(f"[{cat}] ERROR: {e}\n")
    else:
        res = fit_mnl(df, category=args.category, variant=args.variant,
                      outside_good=outside_good)
        print_results(res, label=args.category or "All categories")
