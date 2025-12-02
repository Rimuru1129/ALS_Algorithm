

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


USE_RICH = False
USE_TABULATE = False
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    USE_RICH = True
except Exception:
    try:
        from tabulate import tabulate
        USE_TABULATE = True
    except Exception:
        pass


import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="ALS matrix factorization (exercise-compliant).")
    p.add_argument("--csv", type=str, default="./matrix_1000x200_sparse40.csv", help="Input CSV path")
    p.add_argument("--out-dir", type=str, default="./prediction_csv", help="Output directory")
    p.add_argument("--data-python-dir", type=str, default="./data_python", help="Dir to save reproducer script")
    p.add_argument("-k", type=int, default=10, help="Latent factors")
    p.add_argument("--lambda-reg", type=float, default=0.1, help="Regularization strength")
    p.add_argument("--n-iters", type=int, default=20, help="Maximum ALS iterations (you can set 2000)")
    p.add_argument("--tol", type=float, default=1e-5, help="Early stopping tolerance on relative SSE change (set 0 to disable)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

def print_table(df, title=None, max_rows=20):
    if title:
        print(f"\n--- {title} ---")
    if USE_RICH:
        table = Table(show_header=True, header_style="bold magenta")
        for col in df.columns:
            table.add_column(str(col))
        for _, row in df.head(max_rows).iterrows():
            table.add_row(*[str(x) for x in row.tolist()])
        console.print(table)
    elif USE_TABULATE:
        print(tabulate(df.head(max_rows), headers="keys", tablefmt="psql", showindex=False))
    else:
        print(df.head(max_rows).to_string(index=False))

def als_factorize(R, mask, k=10, lambda_reg=0.1, n_iters=100, tol=0.0, verbose=True, lambda_bias=None):
    
    if lambda_bias is None:
        lambda_bias = lambda_reg

    n_items, n_users = R.shape

    obs = mask == 1
    if obs.sum() == 0:
        raise ValueError("No observed entries in R/mask.")
    mu = R[obs].mean()

    rng = np.random.default_rng()
    V = 0.1 * rng.standard_normal((n_items, k))
    U = 0.1 * rng.standard_normal((n_users, k))


    b_i = np.zeros(n_items, dtype=float)
    b_u = np.zeros(n_users, dtype=float)

    I_k = np.eye(k, dtype=float)

    def compute_sse(R, mask, V, U, b_i, b_u, mu):
        pred = (mu + b_i[:, None] + b_u[None, :] + V @ U.T)
        err = (mask * (R - pred))**2
        return err.sum()

    sse_history = []
    prev_sse = None

    for it in range(1, n_iters + 1):
        t0 = time.time()


        for u in range(n_users):
            idx_items = np.where(mask[:, u] == 1)[0]
            if idx_items.size == 0:
                continue
            V_i = V[idx_items, :]

            r_u = R[idx_items, u] - mu - b_i[idx_items] - b_u[u]
            A = V_i.T @ V_i + lambda_reg * I_k
            b = V_i.T @ r_u
            U[u, :] = np.linalg.solve(A, b)


        for i in range(n_items):
            idx_users = np.where(mask[i, :] == 1)[0]
            if idx_users.size == 0:
                continue
            U_u = U[idx_users, :]
            r_i = R[i, idx_users] - mu - b_u[idx_users] - b_i[i]
            A = U_u.T @ U_u + lambda_reg * I_k
            b = U_u.T @ r_i
            V[i, :] = np.linalg.solve(A, b)


        for i in range(n_items):
            idx_users = np.where(mask[i, :] == 1)[0]
            if idx_users.size == 0:
                b_i[i] = 0.0
                continue

            pred_without_bi = mu + b_u[idx_users] + V[i, :] @ U[idx_users, :].T
            r_minus = R[i, idx_users] - pred_without_bi

            b_i[i] = r_minus.sum() / (idx_users.size + lambda_bias)


        for u in range(n_users):
            idx_items = np.where(mask[:, u] == 1)[0]
            if idx_items.size == 0:
                b_u[u] = 0.0
                continue
            pred_without_bu = mu + b_i[idx_items] + (V[idx_items, :] @ U[u, :])
            r_minus = R[idx_items, u] - pred_without_bu
            b_u[u] = r_minus.sum() / (idx_items.size + lambda_bias)

        sse = compute_sse(R, mask, V, U, b_i, b_u, mu)
        sse_history.append(sse)
        took = time.time() - t0
        if verbose:
            print(f"Iter {it}/{n_iters} - SSE: {sse:.6f} (time: {took:.2f}s)")
        if tol > 0 and prev_sse is not None:
            rel_change = abs(prev_sse - sse) / (prev_sse + 1e-12)
            if rel_change < tol:
                if verbose:
                    print(f"Early stopping at iter {it} (relative SSE change {rel_change:.3e} < tol {tol})")
                break
        prev_sse = sse

    return V, U, b_i, b_u, mu, sse_history

def main():
    args = parse_args()
    np.random.seed(args.seed)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    data_python_dir = Path(args.data_python_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_python_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")


    R = pd.read_csv(csv_path, header=None).values.astype(float)
    n_items, n_users = R.shape
    print(f"Loaded matrix {csv_path} shape {R.shape} (items x users)")

    mask = (R != 0).astype(float)


    start = time.time()
    V, U, b_i, b_u, mu, sse_history = als_factorize(R, mask, k=args.k, lambda_reg=args.lambda_reg, n_iters=args.n_iters, tol=args.tol, verbose=True, lambda_bias=args.lambda_reg)

    total_time = time.time() - start
    print(f"ALS completed in {total_time:.2f}s; iterations run: {len(sse_history)}")


    pred_full = mu + b_i[:, None] + b_u[None, :] + V @ U.T
    pred_rounded = np.round(pred_full * 2.0) / 2.0
    pred_rounded = np.clip(pred_rounded, 1.0, 5.0)



    sse_cont = ((mask * (R - pred_full))**2).sum()
    sse_rounded = ((mask * (R - pred_rounded))**2).sum()
    rmse_cont = np.sqrt(sse_cont / (mask.sum() + 1e-12))
    rmse_rounded = np.sqrt(sse_rounded / (mask.sum() + 1e-12))

    print("\n--- Summary ---")
    print(f"k = {args.k}, lambda = {args.lambda_reg}, iterations run = {len(sse_history)}")
    print(f"SSE (continuous predictions): {sse_cont:.6f}, RMSE: {rmse_cont:.6f}")
    print(f"SSE (rounded -> nearest 0.5, clipped [1,5]): {sse_rounded:.6f}, RMSE: {rmse_rounded:.6f}")


    out_pred_csv = out_dir / "predicted_matrix_rounded.csv"
    pd.DataFrame(pred_rounded).to_csv(out_pred_csv, index=False, header=False)
    print(f"Saved rounded predicted matrix: {out_pred_csv}")


    df_sse = pd.DataFrame({"iteration": list(range(1, len(sse_history) + 1)), "sse": sse_history})
    out_sse_csv = out_dir / "sse_history.csv"
    df_sse.to_csv(out_sse_csv, index=False)
    print(f"Saved SSE history CSV: {out_sse_csv}")


    obs_idx = np.argwhere(mask == 1)
    sample_n = min(len(obs_idx), 25)
    if sample_n > 0:
        rnd = np.random.choice(len(obs_idx), size=sample_n, replace=False)
        sample_pairs = obs_idx[rnd]
        sample_rows = []
        for (i, j) in sample_pairs:
            sample_rows.append({
                "item": int(i), "user": int(j),
                "original": float(R[i, j]),
                "pred_continuous": float(pred_full[i, j]),
                "pred_rounded": float(pred_rounded[i, j])
            })
        df_sample = pd.DataFrame(sample_rows)
        out_sample_csv = out_dir / "sample_original_vs_predictions.csv"
        df_sample.to_csv(out_sample_csv, index=False)
        print(f"Saved sample rows CSV: {out_sample_csv}")
        print_table(df_sample, "Sample original vs predictions (random sample)")
    else:
        print("No observed entries found in mask (unexpected).")


    try:

        plt.figure(figsize=(8, 4.5))
        iters = list(range(1, len(sse_history) + 1))
        plt.plot(iters, sse_history, marker='o', linewidth=1)
        plt.title("SSE on observed entries per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
        sse_plot_path = out_dir / "sse_history.png"
        plt.tight_layout()
        plt.savefig(sse_plot_path, dpi=150)
        plt.close()
        print(f"Saved SSE plot: {sse_plot_path}")


        observed_pred_values = pred_rounded[mask == 1].flatten()
        plt.figure(figsize=(7, 4))
        bins = np.arange(1.0, 5.1, 0.5)
        plt.hist(observed_pred_values, bins=bins, edgecolor='black', align='left')
        plt.title("Histogram of rounded predicted ratings (observed entries)")
        plt.xlabel("Rounded rating")
        plt.ylabel("Count")
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.6)
        hist_path = out_dir / "predicted_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"Saved prediction histogram: {hist_path}")


        obs_idx = np.argwhere(mask == 1)
        n_obs = obs_idx.shape[0]
        if n_obs > 0:
            sample_max = 2000
            sample_size = min(sample_max, n_obs)
            rnd_idx = np.random.choice(n_obs, size=sample_size, replace=False)
            sample_pairs = obs_idx[rnd_idx]
            originals = np.array([R[i, j] for i, j in sample_pairs])
            preds_cont = np.array([pred_full[i, j] for i, j in sample_pairs])

            plt.figure(figsize=(6, 6))
            plt.scatter(originals, preds_cont, s=8, alpha=0.4)
            mn = min(originals.min(), preds_cont.min())
            mx = max(originals.max(), preds_cont.max())
            pad = 0.2
            plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle='--', linewidth=1, color='gray')
            plt.title(f"Original vs continuous prediction (sample n={sample_size})")
            plt.xlabel("Original rating")
            plt.ylabel("Continuous predicted rating")
            plt.xlim(1 - 0.5, 5 + 0.5)
            plt.ylim(1 - 0.5, 5 + 0.5)
            plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
            scatter_path = out_dir / "original_vs_predicted_scatter.png"
            plt.tight_layout()
            plt.savefig(scatter_path, dpi=150)
            plt.close()
            print(f"Saved original vs predicted scatter: {scatter_path}")
    except Exception as e:
        print("Plotting failed:", e)


    reproducer = data_python_dir / "als_solution.py"
    reproducer.write_text(f'''# reproducer script (auto-saved)
# Input CSV: {csv_path}
# Rounding: nearest 0.5, clipped to [1,5]
# k = {args.k}, lambda = {args.lambda_reg}, n_iters = {args.n_iters}
import numpy as np
import pandas as pd
R = pd.read_csv("{csv_path}", header=None).values.astype(float)
mask = (R != 0).astype(float)
# (Reproduce main ALS logic if needed.)
''')
    print(f"Saved reproducer script stub: {reproducer}")

if __name__ == "__main__":
    main()
