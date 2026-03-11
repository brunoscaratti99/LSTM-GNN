import re
import json
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import re
import xarray as xr

def plot_heatmap_nn(M, title="Mapa de calor", cmap="viridis", vmin=None, vmax=None):
    """
    Plota mapa de calor de uma matriz NxN.
    Aceita numpy array, lista de listas ou tensor do PyTorch.
    Retorna (fig, ax).
    """
    if hasattr(M, "detach"):  # torch.Tensor
        M = M.detach().cpu().numpy()
    else:
        M = np.asarray(M)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"A matriz deve ser NxN. Recebido shape={M.shape}")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valor")

    ax.set_title(title)
    ax.set_xlabel("Nó j")
    ax.set_ylabel("Nó i")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))

    plt.tight_layout()
    plt.show()
    return fig, ax




def _parse_lr_token(token):
    if token is None:
        return None
    t = str(token).strip().lower()
    if re.fullmatch(r"e[-+]?\d+", t):  # ex: e-3 -> 1e-3
        t = "1" + t
    try:
        return float(t)
    except ValueError:
        return None
    



def plot_por_hiperparametro_train_val(
    runs_df,
    param="hidden_dim",
    metric_base="mse",                  # "mse", "mae", "loss", "r2", ...
    splits=("train", "val"),            # ("train","val") ou só ("train",) etc
    group_cols=("exp_id", "lr"),
    smooth_window=1,
    show_individual=False,
    min_runs_per_group=1,
    ncols=2,
    figsize_per_plot=(8, 4),
    sharey=False,
):
    if runs_df.empty:
        raise ValueError("runs_df está vazio.")
    if param not in runs_df.columns:
        raise ValueError(f"Coluna '{param}' não existe.")
    if "history" not in runs_df.columns:
        raise ValueError("runs_df precisa da coluna 'history'.")

    values = sorted(runs_df[param].dropna().unique())
    nplots = len(values)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharey=sharey
    )
    axes = np.array(axes).reshape(-1)

    valid_group_cols = [c for c in group_cols if c in runs_df.columns]
    style_map = {"train": "--", "val": "-"}

    for i, pval in enumerate(values):
        ax = axes[i]
        sub = runs_df[(runs_df[param] == pval) & runs_df["history"].notna()].copy()

        if sub.empty:
            ax.set_title(f"{param}={pval} (sem history)")
            ax.grid(True, alpha=0.3)
            continue

        grouped = sub.groupby(valid_group_cols, dropna=False) if valid_group_cols else [(("all",), sub)]
        cmap = plt.get_cmap("tab10")
        gidx = 0

        for gkey, gdf in grouped:
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            color = cmap(gidx % 10)
            gidx += 1

            group_label = " | ".join([f"{c}={v}" for c, v in zip(valid_group_cols, gkey)]) if valid_group_cols else "all"

            for split in splits:
                hkey = f"{split}_{metric_base}"   # ex: train_mse, val_mse
                seqs = []

                for _, row in gdf.iterrows():
                    vals = row["history"].get(hkey)
                    if vals is None or len(vals) == 0:
                        continue

                    arr = np.asarray(vals, dtype=float)
                    if smooth_window > 1:
                        arr = pd.Series(arr).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
                    seqs.append(arr)

                    if show_individual:
                        ax.plot(np.arange(1, len(arr) + 1), arr, alpha=0.10, linewidth=1, color=color, linestyle=style_map.get(split, "-"))

                if len(seqs) < min_runs_per_group:
                    continue

                max_len = max(len(s) for s in seqs)
                M = np.full((len(seqs), max_len), np.nan)
                for r, s in enumerate(seqs):
                    M[r, :len(s)] = s

                mean = np.nanmean(M, axis=0)
                std = np.nanstd(M, axis=0)
                x = np.arange(1, max_len + 1)

                label = f"{group_label} | {split} (n={len(seqs)})"
                line = ax.plot(x, mean, linewidth=2.2, color=color, linestyle=style_map.get(split, "-"), label=label)[0]
                ax.fill_between(x, mean - std, mean + std, alpha=0.14, color=line.get_color())

        ax.set_title(f"{param}={pval}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_base)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{metric_base} por {param} (train/val)", y=1.02)
    plt.tight_layout()
    return fig, axes


def infer_run_metadata(hist_path):
    hist_path = Path(hist_path)
    run_dir = hist_path.parent
    cfg_dir = run_dir.parent
    cfg_name = cfg_dir.name

    meta = {
        "hist_path": str(hist_path),
        "run_dir": str(run_dir),
        "config_name": cfg_name,
        "seed": None,
        "train_period": None,
        "hidden_dim": None,
        "lr": None,
        "dropout_tag": None,   # ex: "05" de +drop_05
        "batch_size": None,
    }

    m = re.search(r"BATCH_TEST_(\d+)", run_dir.name, flags=re.IGNORECASE)
    if m:
        meta["seed"] = int(m.group(1))

    m = re.search(r"(\d+)_DAYS", cfg_name, flags=re.IGNORECASE)
    if m:
        meta["train_period"] = int(m.group(1))

    m = re.search(r"HD_(\d+)", cfg_name, flags=re.IGNORECASE)
    if m:
        meta["hidden_dim"] = int(m.group(1))

    m = re.search(r"LR_([^_+]+)", cfg_name, flags=re.IGNORECASE)
    if m:
        meta["lr"] = _parse_lr_token(m.group(1))

    m = re.search(r"drop_([0-9]+)", cfg_name, flags=re.IGNORECASE)
    if m:
        meta["dropout_tag"] = m.group(1)

    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        meta["batch_size"] = summary.get("batch_size")
        if meta["lr"] is None:
            meta["lr"] = summary.get("lr")

    return meta


def collect_runs(root_dir, recursive=True):
    root = Path(root_dir)
    pattern = "**/hist.pt" if recursive else "hist.pt"

    rows = []
    for hist_fp in root.glob(pattern):
        try:
            hist = torch.load(hist_fp, map_location="cpu")
            if not isinstance(hist, dict):
                continue
        except Exception as e:
            print(f"[WARN] Falha ao ler {hist_fp}: {e}")
            continue

        row = infer_run_metadata(hist_fp)
        row["history"] = hist
        row["epochs"] = len(hist.get("val_mse", []))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("Nenhuma run encontrada.")
        return df

    df = df.sort_values(["config_name", "seed"], na_position="last").reset_index(drop=True)
    print(f"Runs carregadas: {len(df)}")
    return df


def _smooth(values, window=1):
    arr = np.asarray(values, dtype=float)
    if window is None or window <= 1:
        return arr
    return pd.Series(arr).rolling(window=window, min_periods=1).mean().to_numpy()


def _group_label(value, by):
    if isinstance(by, (list, tuple)):
        return " | ".join([f"{k}={v}" for k, v in zip(by, value)])
    return f"{by}={value}"


def plot_convergence_overlay(
    runs_df,
    group_by,
    metric="val_mse",
    include_individual_runs=True,
    smooth_window=1,
    min_runs_per_group=1,
    figsize=(12, 6),
    title=None,
    save_path=None,
):
    if runs_df.empty:
        raise ValueError("runs_df está vazio.")

    by_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    missing = [c for c in by_cols if c not in runs_df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em group_by: {missing}")

    df = runs_df.dropna(subset=by_cols).copy()
    if df.empty:
        raise ValueError("Sem runs com esses hiperparâmetros.")

    groups = []
    for key, g in df.groupby(by_cols, dropna=False):
        if len(g) >= min_runs_per_group:
            groups.append((key, g))
    groups = sorted(groups, key=lambda x: str(x[0]))
    if not groups:
        raise ValueError("Nenhum grupo atende min_runs_per_group.")

    fig, ax = plt.subplots(figsize=figsize)

    for key, g in groups:
        seqs = []
        for _, row in g.iterrows():
            hist = row["history"]
            vals = hist.get(metric, None)
            if vals is None or len(vals) == 0:
                continue
            vals = _smooth(vals, window=smooth_window)
            seqs.append(np.asarray(vals, dtype=float))

            if include_individual_runs:
                x = np.arange(1, len(vals) + 1)
                ax.plot(x, vals, alpha=0.15, linewidth=1)

        if not seqs:
            continue

        max_len = max(len(s) for s in seqs)
        M = np.full((len(seqs), max_len), np.nan, dtype=float)
        for i, s in enumerate(seqs):
            M[i, :len(s)] = s

        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(1, max_len + 1)

        label = f"{_group_label(key, by_cols if len(by_cols) > 1 else by_cols[0])} (n={len(seqs)})"
        line = ax.plot(x, mean, linewidth=2.5, label=label)[0]
        ax.fill_between(x, mean - std, mean + std, alpha=0.20, color=line.get_color())

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Convergência por {group_by} ({metric})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180)

    return fig, ax




def plot_por_hiperparametro_train_val(
    runs_df,
    param="hidden_dim",
    metric_base="mse",                  # "mse", "mae", "loss", "r2", ...
    splits=("train", "val"),            # ("train","val") ou só ("train",) etc
    group_cols=("exp_id", "lr"),
    smooth_window=1,
    show_individual=False,
    min_runs_per_group=1,
    ncols=2,
    figsize_per_plot=(20,10),
    sharey=False,
):
    if runs_df.empty:
        raise ValueError("runs_df está vazio.")
    if param not in runs_df.columns:
        raise ValueError(f"Coluna '{param}' não existe.")
    if "history" not in runs_df.columns:
        raise ValueError("runs_df precisa da coluna 'history'.")

    values = sorted(runs_df[param].dropna().unique())
    nplots = len(values)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharey=sharey
    )
    axes = np.array(axes).reshape(-1)

    valid_group_cols = [c for c in group_cols if c in runs_df.columns]
    style_map = {"train": "--", "val": "-"}

    for i, pval in enumerate(values):
        ax = axes[i]
        sub = runs_df[(runs_df[param] == pval) & runs_df["history"].notna()].copy()

        if sub.empty:
            ax.set_title(f"{param}={pval} (sem history)")
            ax.grid(True, alpha=0.3)
            continue

        grouped = sub.groupby(valid_group_cols, dropna=False) if valid_group_cols else [(("all",), sub)]
        cmap = plt.get_cmap("tab10")
        gidx = 0

        for gkey, gdf in grouped:
            gkey = gkey if isinstance(gkey, tuple) else (gkey,)
            color = cmap(gidx % 10)
            gidx += 1

            group_label = " | ".join([f"{c}={v}" for c, v in zip(valid_group_cols, gkey)]) if valid_group_cols else "all"

            for split in splits:
                hkey = f"{split}_{metric_base}"   # ex: train_mse, val_mse
                seqs = []

                for _, row in gdf.iterrows():
                    vals = row["history"].get(hkey)
                    if vals is None or len(vals) == 0:
                        continue

                    arr = np.asarray(vals, dtype=float)
                    if smooth_window > 1:
                        arr = pd.Series(arr).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
                    seqs.append(arr)

                    if show_individual:
                        ax.plot(np.arange(1, len(arr) + 1), arr, alpha=0.10, linewidth=1, color=color, linestyle=style_map.get(split, "-"))

                if len(seqs) < min_runs_per_group:
                    continue

                max_len = max(len(s) for s in seqs)
                M = np.full((len(seqs), max_len), np.nan)
                for r, s in enumerate(seqs):
                    M[r, :len(s)] = s

                mean = np.nanmean(M, axis=0)
                std = np.nanstd(M, axis=0)
                x = np.arange(1, max_len + 1)

                label = f"{group_label} | {split} (n={len(seqs)})"
                line = ax.plot(x, mean, linewidth=2.2, color=color, linestyle=style_map.get(split, "-"), label=label)[0]
                ax.fill_between(x, mean - std, mean + std, alpha=0.14, color=line.get_color())

        ax.set_title(f"{param}={pval}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_base)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center left',
                  bbox_to_anchor=(1.02,0.5), frameon=False)
        #ax.legend(fontsize=8, loc="upper center", ncol=3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{metric_base} por {param} (train/val)", y=1.02)
    plt.tight_layout()
    return fig, axes



def analisar_experimentos(
    root=r"C:\Experiments",
    group_cols=("window_tag", "exp_id", "train_period", "hidden_dim", "lr", "weight_decay", "batch_size"),
    load_hist=True,
    smooth_window=1,
    show_individual=False,
):
    root = Path(root)

    def _collect_runs():
        rows = []
        for summary_path in root.rglob("run_summary.json"):
            run_dir = summary_path.parent
            rel = run_dir.relative_to(root)
            parts = list(rel.parts)

            window_tag = next((p for p in parts if p.upper().startswith("WINDOW_")), None)
            exp_id = next((p for p in parts if re.match(r"^exp[_-]?\d+$", p, flags=re.IGNORECASE)), None)

            m_batch = re.search(r"BATCH_TEST_(\d+)", run_dir.name, flags=re.IGNORECASE)
            seed = int(m_batch.group(1)) if m_batch else None

            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)

            # trata typo comum: hiddem_dim
            hidden_dim = s.get("hidden_dim", s.get("hiddem_dim"))

            window_days = None
            if window_tag:
                m_win = re.search(r"(\d+)D", window_tag, flags=re.IGNORECASE)
                if m_win:
                    window_days = int(m_win.group(1))

            row = {
                "summary_path": str(summary_path),
                "run_dir": str(run_dir),
                "window_tag": window_tag,
                "window_days": window_days,
                "exp_id": exp_id,
                "seed": seed,
                "train_period": s.get("train_period"),
                "hidden_dim": hidden_dim,
                "lr": s.get("lr"),
                "weight_decay": s.get("weight_decay"),
                "batch_size": s.get("batch_size"),
                "patience": s.get("patience"),
                "epochs_requested": s.get("epochs_requested"),
                "epochs_ran": s.get("epochs_ran"),
                "best_val_loss": s.get("best_val_loss"),
                "last_val_mse": s.get("last_val_mse"),
                "last_val_mae": s.get("last_val_mae"),
                "last_val_r2": s.get("last_val_r2"),
                "last_val_mape": s.get("last_val_mape"),
                "total_time_s": s.get("total_time_s"),
                "history": None,
            }

            if load_hist:
                hist_path = run_dir / "hist.pt"
                if hist_path.exists():
                    hist = torch.load(hist_path, map_location="cpu")
                    row["history"] = hist
                    if "val_mse" in hist and len(hist["val_mse"]) > 0:
                        row["min_val_mse"] = float(np.min(hist["val_mse"]))
                    if "val_mae" in hist and len(hist["val_mae"]) > 0:
                        row["min_val_mae"] = float(np.min(hist["val_mae"]))
                    if "val_r2" in hist and len(hist["val_r2"]) > 0:
                        row["max_val_r2"] = float(np.max(hist["val_r2"]))

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values(["window_tag", "exp_id", "seed"], na_position="last").reset_index(drop=True)

    def _build_tables(runs_df):
        if runs_df.empty:
            return runs_df, runs_df

        per_run_cols = [
            "window_tag", "exp_id", "seed",
            "train_period", "hidden_dim", "lr", "weight_decay", "batch_size",
            "epochs_ran", "best_val_loss", "last_val_mse", "last_val_mae", "last_val_r2",
            "min_val_mse", "min_val_mae", "max_val_r2", "total_time_s",
        ]
        per_run_cols = [c for c in per_run_cols if c in runs_df.columns]
        tabela_runs = runs_df[per_run_cols].copy()

        gcols = [c for c in group_cols if c in runs_df.columns]

        agg_spec = {
            "seed": "count",
            "epochs_ran": ["mean", "std"],
            "best_val_loss": ["mean", "std", "min"],
            "last_val_mse": ["mean", "std"],
            "last_val_mae": ["mean", "std"],
            "last_val_r2": ["mean", "std", "max"],
            "min_val_mse": ["mean", "std"],
            "min_val_mae": ["mean", "std"],
            "max_val_r2": ["mean", "std"],
            "total_time_s": ["mean", "sum"],
        }
        agg_spec = {k: v for k, v in agg_spec.items() if k in runs_df.columns}

        tabela_exp = runs_df.groupby(gcols, dropna=False).agg(agg_spec)
        tabela_exp.columns = ["_".join([x for x in col if x]).strip("_") for col in tabela_exp.columns.to_flat_index()]
        tabela_exp = tabela_exp.rename(columns={"seed_count": "n_runs"}).reset_index()

        if "best_val_loss_mean" in tabela_exp.columns:
            tabela_exp = tabela_exp.sort_values("best_val_loss_mean", ascending=True)

        if "total_time_s_sum" in tabela_exp.columns:
            tabela_exp["total_time_h_sum"] = tabela_exp["total_time_s_sum"] / 3600.0

        return tabela_runs, tabela_exp

    def _plot_metric(runs_df, metric="val_mse", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 5))
        else:
            fig = ax.figure

        gcols = [c for c in group_cols if c in runs_df.columns]
        plot_df = runs_df[runs_df["history"].notna()].copy()
        if plot_df.empty:
            ax.set_title(f"{metric} (sem hist.pt)")
            return fig, ax

        for key, g in plot_df.groupby(gcols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            seqs = []
            for _, row in g.iterrows():
                vals = row["history"].get(metric)
                if vals is None or len(vals) == 0:
                    continue
                arr = np.asarray(vals, dtype=float)
                if smooth_window > 1:
                    arr = pd.Series(arr).rolling(window=smooth_window, min_periods=1).mean().to_numpy()
                seqs.append(arr)
                if show_individual:
                    ax.plot(np.arange(1, len(arr) + 1), arr, alpha=0.12, linewidth=1)

            if not seqs:
                continue

            max_len = max(len(s) for s in seqs)
            M = np.full((len(seqs), max_len), np.nan)
            for i, s in enumerate(seqs):
                M[i, :len(s)] = s

            mean = np.nanmean(M, axis=0)
            std = np.nanstd(M, axis=0)
            x = np.arange(1, max_len + 1)

            label = " | ".join([f"{c}={v}" for c, v in zip(gcols, key)]) + f" (n={len(seqs)})"
            line = ax.plot(x, mean, linewidth=2.2, label=label)[0]
            ax.fill_between(x, mean - std, mean + std, alpha=0.16, color=line.get_color())

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        return fig, ax

    runs_df = _collect_runs()
    tabela_runs, tabela_exp = _build_tables(runs_df)

    # painel padrão
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["val_mse", "val_mae", "val_r2"]):
        _plot_metric(runs_df, metric=metric, ax=ax)

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    return runs_df, tabela_runs, tabela_exp, (fig, axes)



def save_error_plots(path, train_mse, val_mse, train_mae, val_mae, train_r2, val_r2):
    metrics = [
        ("mse", train_mse, val_mse, "MSE"),
        ("mae", train_mae, val_mae, "MAE"),
        ("r2", train_r2, val_r2, "R2"),
    ]

    for metric_name, train_values, val_values, y_label in metrics:
        plt.figure(figsize=(12, 6))
        if len(train_values) > 0:
            plt.plot(train_values, label="train", linewidth=2)
        if len(val_values) > 0:
            plt.plot(val_values, label="val", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.title(f"{y_label} by epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{metric_name}_curve.png"), dpi=150)
        plt.close()
        
        
def model_weights_hist(model):
    fig, axs = plt.subplots(1,3, figsize=(12,3))
    i = 1
    for name, param in model.named_parameters():
        axs[i-1].hist(param.detach().cpu().flatten(), bins=100)
        axs[i-1].set_title(name)
        axs[i-1].set_xlim(-1e0, 1e0)
        if i%3==0:
            plt.tight_layout()
            plt.show()
            fig, axs = plt.subplots(1,3, figsize=(12,3))
            i = 1
        else:
            i += 1