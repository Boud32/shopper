"""
Multinomial Logit (MNL) estimation for the Shopper experiment engine.

Linear utility of product j in experiment n (log-odds scale):

    U_j = beta_0 + beta_pos * position_j + beta_price * price_j
        + beta_rating * rating_j + beta_logrev * log(1 + review_count_j)
        + beta_sp * is_sponsored_j + beta_bs * is_best_seller_j
        + beta_op * is_overall_pick_j

The softmax weight is V_j = exp(U_j). Choice probabilities:

  outside_good=True  — outside good included, beta_0 estimated.
      Requires ~30% no_purchase rate for beta_0 to be identified.
      U_no_purchase fixed at 0, so V_no_purchase = 1 in the denominator.
      P(j) = V_j / (1 + sum_k V_k)
      P(no_purchase) = 1 / (1 + sum_k V_k)

  outside_good=False — no outside good, beta_0 dropped.
      no_purchase experiments are excluded.
      P(j) = V_j / sum_k V_k

Continuous features are normalized (mean 0, std 1) inside prepare_experiments
so coefficients are on a comparable scale. Raw means and stds are stored in
norm_params and printed when verbose=True.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CONT_FEATURES_LOG = ["position", "price", "rating", "log_review_count"]
CONT_FEATURES_RAW = ["position", "price", "rating", "review_count"]
BIN_FEATURES      = ["is_sponsored", "is_best_seller", "is_overall_pick"]
FEATURES          = CONT_FEATURES_LOG + BIN_FEATURES  # default (log spec)


def prepare_experiments(df, category=None, variant="full", outside_good=True,
                        log_reviews=True):
    """
    log_reviews=True  — use log(1 + review_count) as the reviews feature (default).
    log_reviews=False — use raw review_count instead.
    """
    sub = df[df["variant"] == variant].copy()
    if category:
        sub = sub[sub["category"] == category]

    sub = sub.dropna(subset=["price", "rating", "review_count", "position"])
    if sub.empty:
        raise ValueError("No rows remain after filtering — check category/variant.")

    sub["log_review_count"] = np.log1p(sub["review_count"].astype(float))

    cont_features = CONT_FEATURES_LOG if log_reviews else CONT_FEATURES_RAW
    all_features  = cont_features + BIN_FEATURES

    norm_params = {}
    for col in cont_features:
        mu    = sub[col].mean()
        sigma = sub[col].std()
        norm_params[col] = {"mu": mu, "sigma": sigma}
        sub[col] = (sub[col] - mu) / max(sigma, 1e-10)

    for col in BIN_FEATURES:
        sub[col] = sub[col].astype(float)

    experiments = []
    for _, group in sub.groupby("experiment_id"):
        X           = group[all_features].values.astype(float)
        chosen_mask = group["chosen"].values
        chosen_idx  = int(np.where(chosen_mask)[0][0]) if chosen_mask.any() else None

        if not outside_good and chosen_idx is None:
            continue

        experiments.append({"X": X, "chosen_idx": chosen_idx})

    return experiments, norm_params, all_features


def _nll(betas, experiments, outside_good):
    """Objective for L-BFGS-B: negative log-likelihood of the MNL choice over all experiments. Gradient is estimated numerically by scipy."""
    beta_0    = betas[0]  if outside_good else 0.0
    beta_rest = betas[1:] if outside_good else betas

    total_nll = 0.0
    for exp in experiments:
        X          = exp["X"]
        chosen_idx = exp["chosen_idx"]
        utilities  = beta_0 + X @ beta_rest
        exp_u      = np.exp(utilities)
        sum_exp    = np.sum(exp_u)
        denom      = (1.0 + sum_exp) if outside_good else sum_exp
        prob       = (1.0 / denom) if chosen_idx is None else (exp_u[chosen_idx] / denom)
        total_nll -= np.log(prob + 1e-300)

    return total_nll


def log_likelihood(betas, experiments, outside_good=True):
    return _nll(betas, experiments, outside_good)


def compute_utilities(experiments, betas, outside_good=True):
    """Return per-experiment U_j (linear utility, log-odds) arrays given estimated betas. Exponentiate for V_j."""
    beta_0    = betas[0]  if outside_good else 0.0
    beta_rest = betas[1:] if outside_good else betas

    return [
        {"utilities": beta_0 + exp["X"] @ beta_rest, "chosen_idx": exp["chosen_idx"]}
        for exp in experiments
    ]


def predicted_no_purchase_rate(experiments, betas, outside_good=True):
    """
    Model-implied mean P(no_purchase) given fitted betas.

    At the MLE this equals the observed no-purchase rate — a large discrepancy
    means the optimizer did not converge to the true MLE.
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


def fit_mnl(df, category=None, variant="full", outside_good=True, verbose=False,
            log_reviews=True):
    experiments, norm_params, all_features = prepare_experiments(
        df, category=category, variant=variant, outside_good=outside_good,
        log_reviews=log_reviews,
    )

    feature_names  = (["beta_0"] + all_features) if outside_good else all_features
    initial_betas  = np.zeros(len(feature_names))
    n_exp          = len(experiments)
    n_no_purchase  = sum(1 for e in experiments if e["chosen_idx"] is None)

    if outside_good:
        ll_null = sum(np.log(1.0 / (len(exp["X"]) + 1)) for exp in experiments)
    else:
        ll_null = sum(np.log(1.0 / len(exp["X"])) for exp in experiments)

    if verbose:
        print(f"── Data {'─'*45}")
        print(f"  Experiments : {n_exp:,}  "
              f"(no-purchase: {n_no_purchase} = {100 * n_no_purchase / n_exp:.1f}%)")
        print(f"\n  {'Feature':<20}  {'mean (raw)':>10}  {'std (raw)':>10}")
        print(f"  {'─'*20}  {'─'*10}  {'─'*10}")
        for feat in norm_params:
            p = norm_params[feat]
            print(f"  {feat:<20}  {p['mu']:>10.3f}  {p['sigma']:>10.3f}")
        rc_sub = df[df["variant"] == variant].copy()
        if category:
            rc_sub = rc_sub[rc_sub["category"] == category]
        rc = rc_sub["review_count"].dropna().astype(float)
        p95 = rc.quantile(0.95)
        dominated_share = rc[rc > p95].sum() / rc.sum() * 100
        print(f"\n  {'review_count distribution':<20}")
        print(f"     median : {rc.median():>10,.0f}")
        print(f"     mean   : {rc.mean():>10,.0f}")
        print(f"     p95    : {p95:>10,.0f}")
        print(f"     p99    : {rc.quantile(0.99):>10,.0f}")
        print(f"     max    : {rc.max():>10,.0f}")
        print(f"     top 5% account for {dominated_share:.0f}% of total review mass")
        init_nll = _nll(initial_betas, experiments, outside_good)
        print(f"\n  NLL at β=0  : {init_nll:.1f}  (null LL: {-ll_null:.1f})")
        print(f"\n── Optimizing (L-BFGS-B) {'─'*27}")

    counter = [0]

    def _callback(betas):
        counter[0] += 1
        if verbose and counter[0] % 10 == 0:
            nll = _nll(betas, experiments, outside_good)
            print(f"  iter {counter[0]:4d}  NLL = {nll:.3f}")

    result = minimize(
        _nll,
        x0=initial_betas,
        args=(experiments, outside_good),
        method="L-BFGS-B",
        jac=False,
        callback=_callback,
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
    )

    beta_hat  = result.x
    ll        = -result.fun
    pseudo_r2 = 1.0 - ll / ll_null
    grad_norm = float(np.linalg.norm(result.jac)) if result.jac is not None else None

    return {
        "summary":        pd.DataFrame({"feature": feature_names, "coef": beta_hat}),
        "log_likelihood": ll,
        "ll_null":        ll_null,
        "pseudo_r2":      pseudo_r2,
        "n_experiments":  n_exp,
        "n_no_purchase":  n_no_purchase,
        "outside_good":   outside_good,
        "converged":      result.success,
        "grad_norm":      grad_norm,
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
    if res.get("grad_norm") is not None:
        print(f"  Gradient norm : {res['grad_norm']:.2e}")
    if res["outside_good"]:
        pred_np, _ = predicted_no_purchase_rate(
            res["experiments"], res["beta"], outside_good=True
        )
        obs_np = res["n_no_purchase"] / res["n_experiments"]
        print(f"  Pred no-purch : {pred_np * 100:.1f}%  (observed: {obs_np * 100:.1f}%)")
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
    parser.add_argument("--results-dir",     default="data/architecture2/results")
    args = parser.parse_args()

    outside_good = not args.no_outside_good
    df = load_results_to_dataframe(results_dir=args.results_dir)

    if args.all_categories:
        for cat in sorted(df["category"].dropna().unique()):
            try:
                res = fit_mnl(df, category=cat, variant=args.variant,
                              outside_good=outside_good, verbose=True)
                print_results(res, label=cat)
            except Exception as e:
                print(f"[{cat}] ERROR: {e}\n")
    else:
        res = fit_mnl(df, category=args.category, variant=args.variant,
                      outside_good=outside_good, verbose=True)
        print_results(res, label=args.category or "All categories")
