"""
Multinomial Logit (MNL) estimation for the Shopper experiment engine.

Utility of product j in experiment n:

    u_j = beta_0
          + beta_pos      * position_j
          + beta_price    * price_j
          + beta_rating   * rating_j
          + beta_logrev   * log(1 + review_count_j)
          + beta_sp       * is_sponsored_j
          + beta_bs       * is_best_seller_j
          + beta_op       * is_overall_pick_j

Outside good (no_purchase) utility is normalized to 0, so exp(0) = 1.
beta_0 is the intercept: baseline utility of buying *anything* vs. not buying.
Because the outside good utility is fixed at 0, beta_0 is identified.

Choice probability:
    P(j | S)           = exp(u_j) / (1 + sum_{k in S} exp(u_k))
    P(no_purchase | S) = 1        / (1 + sum_{k in S} exp(u_k))

Estimation: maximise log-likelihood via scipy.optimize.minimize (Nelder-Mead).
No analytic gradient or Hessian is used.

Usage:
    from src.analysis_helper import load_results_to_dataframe
    from src.mnl import fit_mnl, print_results

    df  = load_results_to_dataframe()
    res = fit_mnl(df, category="Running Shoes")
    print_results(res, "Running Shoes")
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Continuous features (will be z-scored before fitting)
CONT_FEATURES = ["position", "price", "rating", "log_review_count"]
# Binary features (left as 0/1)
BIN_FEATURES  = ["is_sponsored", "is_best_seller", "is_overall_pick"]
FEATURES      = CONT_FEATURES + BIN_FEATURES


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_experiments(df, category=None, variant="full"):
    """
    Convert the long-format DataFrame into a list of experiment dicts,
    one dict per offer set.

    Each dict has:
        X          ndarray (n_products, n_features)  — feature matrix
        chosen_idx int or None  — index of the chosen product within X,
                                   or None if the experiment ended in no_purchase

    Continuous features are z-scored across all experiments (not per-experiment)
    so that coefficient magnitudes are comparable.
    """
    sub = df[df["variant"] == variant].copy()
    if category:
        sub = sub[sub["category"] == category]

    sub = sub.dropna(subset=["price", "rating", "review_count", "position"])
    if sub.empty:
        raise ValueError("No rows remain after filtering — check category/variant.")

    # Log-transform review count (heavy right skew)
    sub["log_review_count"] = np.log1p(sub["review_count"].astype(float))

    # Z-score continuous features across all rows
    for col in CONT_FEATURES:
        mu    = sub[col].mean()
        sigma = sub[col].std()
        sub[col] = (sub[col] - mu) / max(sigma, 1e-10)

    # Binary features as floats
    for col in BIN_FEATURES:
        sub[col] = sub[col].astype(float)

    # Build one dict per experiment
    experiments = []
    for exp_id, group in sub.groupby("experiment_id"):
        X           = group[FEATURES].values.astype(float)  # (n_products, n_features)
        chosen_mask = group["chosen"].values                  # boolean array

        if chosen_mask.any():
            chosen_idx = int(np.where(chosen_mask)[0][0])   # index of chosen product
        else:
            chosen_idx = None                                # no_purchase

        experiments.append({"X": X, "chosen_idx": chosen_idx})

    return experiments


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def log_likelihood(betas, experiments):
    """
    Negative log-likelihood of the MNL model.

    betas[0]   = beta_0  (intercept — baseline utility of purchasing vs. not)
    betas[1:]  = one coefficient per feature in FEATURES (same order)

    For each experiment n:
      - compute utility u_j = beta_0 + X_j @ beta_rest for every product j
      - if a product was chosen:
            prob = exp(u_chosen) / (1 + sum_j exp(u_j))
      - if no_purchase:
            prob = 1 / (1 + sum_j exp(u_j))
        (the "1" in the numerator is exp(0), the outside good utility)
      - accumulate log(prob) across all experiments
    """
    beta_0    = betas[0]
    beta_rest = betas[1:]

    total_log_lik = 0.0

    for exp in experiments:
        X          = exp["X"]           # (n_products, n_features)
        chosen_idx = exp["chosen_idx"]  # int or None

        # Utility of each product in this offer set
        utilities = beta_0 + X @ beta_rest  # shape: (n_products,)

        # Sum of exp(utility) across all products  — denominates both cases
        sum_exp = np.sum(np.exp(utilities))

        if chosen_idx is None:
            # no_purchase: outside good has utility 0, so exp(0) = 1
            prob = 1.0 / (1.0 + sum_exp)
        else:
            prob = np.exp(utilities[chosen_idx]) / (1.0 + sum_exp)

        total_log_lik += np.log(prob)

    return -total_log_lik   # negative because scipy.minimize minimizes


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_mnl(df, category=None, variant="full"):
    """
    Fit the MNL model and return a results dict.

    Returns:
        dict with keys:
            summary        pd.DataFrame  — feature name and estimated coefficient
            log_likelihood float
            ll_null        float         — log L of uniform model (1 / n_alternatives)
            pseudo_r2      float         — McFadden R² = 1 − LL_full / LL_null
            n_experiments  int
            n_no_purchase  int           — experiments that ended in no_purchase
            converged      bool
            beta           ndarray       — raw coefficient array [beta_0, beta_1, ...]
    """
    experiments = prepare_experiments(df, category=category, variant=variant)

    # Initial guess: all zeros (neutral — no preference for any feature)
    n_params      = 1 + len(FEATURES)   # beta_0 + one per feature
    initial_betas = np.zeros(n_params)

    result = minimize(
        log_likelihood,
        x0=initial_betas,
        args=(experiments,),
        method="Nelder-Mead",
        options={"maxiter": 100_000, "xatol": 1e-6, "fatol": 1e-6},
    )

    beta_hat = result.x

    # Null log-likelihood: uniform choice over (n_products + 1) options
    # — the +1 accounts for the always-available outside good
    ll_null = sum(
        np.log(1.0 / (len(exp["X"]) + 1)) for exp in experiments
    )
    ll        = -result.fun
    pseudo_r2 = 1.0 - ll / ll_null

    feature_names = ["beta_0"] + FEATURES
    summary = pd.DataFrame({
        "feature": feature_names,
        "coef":    beta_hat,
    })

    n_no_purchase = sum(1 for exp in experiments if exp["chosen_idx"] is None)

    return {
        "summary":        summary,
        "log_likelihood": ll,
        "ll_null":        ll_null,
        "pseudo_r2":      pseudo_r2,
        "n_experiments":  len(experiments),
        "n_no_purchase":  n_no_purchase,
        "converged":      result.success,
        "beta":           beta_hat,
    }


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def print_results(res, label=""):
    header = f"MNL Results — {label}" if label else "MNL Results"
    print("=" * 60)
    print(header)
    print("=" * 60)
    print(f"  N experiments : {res['n_experiments']:,}")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.analysis_helper import load_results_to_dataframe

    parser = argparse.ArgumentParser(description="Fit MNL model to Shopper results")
    parser.add_argument("--category",       default=None,  help="Category to filter to")
    parser.add_argument("--variant",        default="full", help="Prompt variant (default: full)")
    parser.add_argument("--all-categories", action="store_true",
                        help="Fit a separate model per category")
    args = parser.parse_args()

    df = load_results_to_dataframe()

    if args.all_categories:
        for cat in sorted(df["category"].dropna().unique()):
            try:
                res = fit_mnl(df, category=cat, variant=args.variant)
                print_results(res, label=cat)
            except Exception as e:
                print(f"[{cat}] ERROR: {e}\n")
    else:
        res = fit_mnl(df, category=args.category, variant=args.variant)
        print_results(res, label=args.category or "All categories")
