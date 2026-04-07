"""
Multinomial Logit (MNL) estimation for the Shopper experiment engine.

Utility of product j in experiment n:

    u_j = beta_0 + beta_pos * position_j + beta_price * price_j
        + beta_rating * rating_j + beta_logrev * log(1 + review_count_j)
        + beta_sp * is_sponsored_j + beta_bs * is_best_seller_j
        + beta_op * is_overall_pick_j

Two specs via outside_good argument:

  outside_good=True  — outside good included, beta_0 estimated.
      Requires ~30% no_purchase rate for beta_0 to be identified.
      P(j) = exp(u_j) / (1 + sum_k exp(u_k))
      P(no_purchase) = 1 / (1 + sum_k exp(u_k))

  outside_good=False — no outside good, beta_0 dropped.
      no_purchase experiments are excluded. Use when no_purchase rate is near zero.
      Coefficients are proportional to the outside_good=True spec under IIA.
      P(j) = exp(u_j) / sum_k exp(u_k)
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

    # log(1 + review_count) to capture diminishing returns to social proof
    sub["log_review_count"] = np.log1p(sub["review_count"].astype(float))

    # normalize continuous features (mean 0, std 1) so coefficients are comparable
    for col in CONT_FEATURES:
        mu    = sub[col].mean()
        sigma = sub[col].std()
        sub[col] = (sub[col] - mu) / max(sigma, 1e-10)

    for col in BIN_FEATURES:
        sub[col] = sub[col].astype(float)

    experiments = []
    for _, group in sub.groupby("experiment_id"):
        X           = group[FEATURES].values.astype(float)
        chosen_mask = group["chosen"].values

        if chosen_mask.any():
            chosen_idx = int(np.where(chosen_mask)[0][0])
        else:
            chosen_idx = None  # no_purchase

        if not outside_good and chosen_idx is None:
            continue

        experiments.append({"X": X, "chosen_idx": chosen_idx})

    return experiments


def log_likelihood(betas, experiments, outside_good=True):
    if outside_good:
        beta_0    = betas[0]
        beta_rest = betas[1:]
    else:
        beta_0    = 0.0
        beta_rest = betas

    total_log_lik = 0.0

    for exp in experiments:
        X          = exp["X"]
        chosen_idx = exp["chosen_idx"]

        utilities = beta_0 + X @ beta_rest
        sum_exp   = np.sum(np.exp(utilities))

        if outside_good:
            denom = 1.0 + sum_exp  # exp(0) = 1 for the outside good
        else:
            denom = sum_exp

        if chosen_idx is None:  # no_purchase
            prob = 1.0 / denom
        else:
            prob = np.exp(utilities[chosen_idx]) / denom

        total_log_lik += np.log(prob)

    return -total_log_lik  # minimize expects a positive value


def fit_mnl(df, category=None, variant="full", outside_good=True):
    experiments = prepare_experiments(df, category=category, variant=variant,
                                      outside_good=outside_good)

    if outside_good:
        feature_names = ["beta_0"] + FEATURES
    else:
        feature_names = FEATURES

    initial_betas = np.zeros(len(feature_names))

    result = minimize(
        log_likelihood,
        x0=initial_betas,
        args=(experiments, outside_good),
        method="Nelder-Mead",
        options={"maxiter": 100_000, "xatol": 1e-6, "fatol": 1e-6},
    )

    beta_hat = result.x

    # null model: uniform choice over all alternatives
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
