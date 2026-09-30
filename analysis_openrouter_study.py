"""
Post-hoc analysis of OpenRouter OCT classification study.

Generates a unified summary figure and additional analyses from the
per-model prediction CSVs saved by openrouter_study.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

CLASSES = ["CNV", "DRUSEN", "DME", "CSR", "MH", "NORMAL"]

MODELS = {
    "GPT-5.4-mini": "openai_gpt-5.4-mini_predictions.csv",
    "Gemini-3-Flash": "google_gemini-3-flash-preview_predictions.csv",
    "Llama-3.2-11B-Vision": "meta-llama_llama-3.2-11b-vision-instruct_predictions.csv",
}

# Models highlighted in the summary figure (Llama's F1 is zero for four of
# the six classes).
FOCUS_MODELS = ["Gemini-3-Flash", "GPT-5.4-mini"]

MODEL_COLORS = {
    "GPT-5.4-mini": "#10a37f",
    "Gemini-3-Flash": "#4285f4",
    "Llama-3.2-11B-Vision": "#f5a623",
}

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "openrouter_study_complete"
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions():
    dfs = {}
    for name, fname in MODELS.items():
        df = pd.read_csv(RESULTS_DIR / fname)
        df["valid"] = df["predicted_label"].isin(CLASSES)
        dfs[name] = df
    return dfs


def overall_metrics(dfs):
    rows = []
    for name, df in dfs.items():
        valid = df[df["valid"]]
        y_true = valid["true_label"].values
        y_pred = valid["predicted_label"].values
        acc = (y_true == y_pred).mean() if len(valid) else 0.0
        macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred, labels=CLASSES) if len(valid) else 0.0
        mcc = matthews_corrcoef(y_true, y_pred) if len(valid) else 0.0
        tokens = int(df["total_tokens"].sum())
        mean_completion = float(df["completion_tokens"].mean())
        rows.append(
            {
                "model": name,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "cohen_kappa": kappa,
                "mcc": mcc,
                "valid": int(df["valid"].sum()),
                "total": len(df),
                "total_tokens": tokens,
                "avg_completion_tokens": mean_completion,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def per_class_f1(dfs):
    records = []
    for name, df in dfs.items():
        valid = df[df["valid"]]
        y_true = valid["true_label"].values
        y_pred = valid["predicted_label"].values
        rep = classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
        )
        for cls in CLASSES:
            records.append(
                {
                    "model": name,
                    "class": cls,
                    "precision": rep[cls]["precision"],
                    "recall": rep[cls]["recall"],
                    "f1": rep[cls]["f1-score"],
                    "support": rep[cls]["support"],
                }
            )
    return pd.DataFrame(records)


def agreement_matrix(dfs):
    """Pairwise Cohen's kappa between model predictions on samples
    where all models produced a valid class label."""
    names = list(dfs.keys())
    merged = None
    for name, df in dfs.items():
        sub = df[["filename", "true_label", "predicted_label", "valid"]].rename(
            columns={"predicted_label": name, "valid": f"{name}_valid"}
        )
        merged = sub if merged is None else merged.merge(sub, on=["filename", "true_label"], how="inner")

    mask = np.ones(len(merged), dtype=bool)
    for name in names:
        mask &= merged[f"{name}_valid"].values
    merged = merged[mask]

    kappa = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            kappa.loc[a, b] = cohen_kappa_score(merged[a].values, merged[b].values, labels=CLASSES)
    return kappa, merged


def consensus_analysis(merged):
    """3-model consensus stats (kept for completeness, not the headline)."""
    names = [c for c in merged.columns if c not in ("filename", "true_label") and not c.endswith("_valid")]
    preds = merged[names].values
    all_agree = np.all(preds == preds[:, :1], axis=1)
    majority = []
    for row in preds:
        vals, counts = np.unique(row, return_counts=True)
        if counts.max() >= 2:
            majority.append(vals[counts.argmax()])
        else:
            majority.append("NO_CONSENSUS")
    majority = np.array(majority)
    true = merged["true_label"].values
    stats = {
        "n_samples": len(merged),
        "all_agree_pct": float(all_agree.mean() * 100),
        "all_agree_correct_pct": float(
            ((preds[all_agree, 0] == true[all_agree]).mean() * 100) if all_agree.any() else 0.0
        ),
        "majority_available_pct": float((majority != "NO_CONSENSUS").mean() * 100),
        "majority_correct_pct": float(
            ((majority[majority != "NO_CONSENSUS"] == true[majority != "NO_CONSENSUS"]).mean() * 100)
            if (majority != "NO_CONSENSUS").any()
            else 0.0
        ),
    }
    return stats, majority, all_agree


def two_model_consensus(dfs, model_a="Gemini-3-Flash", model_b="GPT-5.4-mini"):
    """Pairwise agreement / disagreement statistics between the two
    proprietary models — this is the headline consensus result in the
    abstract."""
    a = dfs[model_a][["filename", "true_label", "predicted_label", "valid"]].rename(
        columns={"predicted_label": model_a, "valid": f"{model_a}_valid"}
    )
    b = dfs[model_b][["filename", "predicted_label", "valid"]].rename(
        columns={"predicted_label": model_b, "valid": f"{model_b}_valid"}
    )
    m = a.merge(b, on="filename")
    m = m[m[f"{model_a}_valid"] & m[f"{model_b}_valid"]]
    n = len(m)
    agree = m[m[model_a] == m[model_b]]
    disagree = m[m[model_a] != m[model_b]]
    stats = {
        "model_a": model_a,
        "model_b": model_b,
        "n": int(n),
        "agreement_rate": float(len(agree) / n),
        "accuracy_when_agree": float((agree[model_a] == agree["true_label"]).mean()),
        "disagreement_rate": float(len(disagree) / n),
        "model_a_accuracy_on_disagreement": float((disagree[model_a] == disagree["true_label"]).mean()),
        "model_b_accuracy_on_disagreement": float((disagree[model_b] == disagree["true_label"]).mean()),
    }
    return stats, agree, disagree


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_summary_figure(summary, pc_df, dfs, kappa, out_path):
    """Main figure focused on the two proprietary models.  Llama is reported
    only as a negative result and not included in the per-class / confusion
    panels."""
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.30)

    # Restrict to the focus (proprietary) models for the main panels
    summary_focus = summary[summary["model"].isin(FOCUS_MODELS)].copy()
    summary_focus = summary_focus.set_index("model").loc[FOCUS_MODELS].reset_index()

    # (a) Overall accuracy + macro-F1 bar chart
    ax = fig.add_subplot(gs[0, 0])
    models_order = summary_focus["model"].tolist()
    x = np.arange(len(models_order))
    width = 0.35
    ax.bar(x - width / 2, summary_focus["accuracy"], width, label="Accuracy", color="#4c72b0")
    ax.bar(x + width / 2, summary_focus["macro_f1"], width, label="Macro-F1", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Overall performance", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    for i, (acc, f1) in enumerate(zip(summary_focus["accuracy"], summary_focus["macro_f1"])):
        ax.text(i - width / 2, acc + 0.01, f"{acc:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f1 + 0.01, f"{f1:.2f}", ha="center", fontsize=9)

    # (b) Cohen's kappa + MCC
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - width / 2, summary_focus["cohen_kappa"], width, label="Cohen's κ", color="#55a868")
    ax.bar(x + width / 2, summary_focus["mcc"], width, label="MCC", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Chance-corrected agreement", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    for i, (k, m) in enumerate(zip(summary_focus["cohen_kappa"], summary_focus["mcc"])):
        ax.text(i - width / 2, k + 0.01, f"{k:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=9)

    # (c) Per-class F1 grouped bar (proprietary models only)
    ax = fig.add_subplot(gs[1, 0])
    pivot = pc_df.pivot(index="class", columns="model", values="f1").reindex(CLASSES)
    pivot = pivot[models_order]
    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS[m] for m in models_order], width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("")
    ax.set_xticklabels(CLASSES, rotation=0)
    ax.set_title("(c) Per-class F1", fontsize=12, fontweight="bold")
    ax.legend(title="Model", loc="upper right", fontsize=9)

    # (d) Class support (dataset balance)
    ax = fig.add_subplot(gs[1, 1])
    support = pc_df[pc_df["model"] == models_order[0]].set_index("class")["support"].reindex(CLASSES)
    ax.bar(CLASSES, support.values, color=sns.color_palette("viridis", len(CLASSES)))
    ax.set_ylabel("# samples")
    ax.set_title("(d) Test-set class distribution", fontsize=12, fontweight="bold")
    for i, v in enumerate(support.values):
        ax.text(i, v + 2, str(int(v)), ha="center", fontsize=9)

    # (e,f) Row-normalized confusion matrices for the focus models
    for i, name in enumerate(models_order):
        ax = fig.add_subplot(gs[2, i])
        valid = dfs[name][dfs[name]["valid"]]
        cm = confusion_matrix(valid["true_label"], valid["predicted_label"], labels=CLASSES)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False,
            annot_kws={"size": 9},
        )
        letter = chr(ord("e") + i)
        ax.set_title(f"({letter}) {name}\nrow-normalized confusion matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("True")
        else:
            ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle(
        "Zero-shot OCT classification with vision-language models (n = 919, 6 classes)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_agreement(dfs, merged, out_path):
    """Heatmap of top confusion pairs for the best model, stratified by
    whether the worst/medium models also got it wrong."""
    best = "Gemini-3-Flash"
    valid = dfs[best][dfs[best]["valid"]]
    errors = valid[valid["true_label"] != valid["predicted_label"]]
    pair_counts = (
        errors.groupby(["true_label", "predicted_label"]).size().reset_index(name="count")
    )
    pair_counts = pair_counts.sort_values("count", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = pair_counts["true_label"] + " -> " + pair_counts["predicted_label"]
    ax.barh(labels[::-1], pair_counts["count"].values[::-1], color="#4285f4")
    ax.set_xlabel("# errors")
    ax.set_title(f"Top confusions for {best}", fontweight="bold")
    for i, v in enumerate(pair_counts["count"].values[::-1]):
        ax.text(v + 0.3, i, str(int(v)), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_histogram(dfs, out_path):
    """For each sample, count how many of the focus (proprietary) models
    got it right -> dataset difficulty."""
    names = FOCUS_MODELS
    merged = None
    for name in names:
        sub = dfs[name][["filename", "true_label", "predicted_label"]].rename(
            columns={"predicted_label": name}
        )
        merged = sub if merged is None else merged.merge(sub, on=["filename", "true_label"])

    correct = np.zeros(len(merged), dtype=int)
    for name in names:
        correct += (merged[name] == merged["true_label"]).astype(int).values
    merged["n_correct"] = correct

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # Global histogram
    counts = merged["n_correct"].value_counts().reindex(range(len(names) + 1), fill_value=0)
    axes[0].bar(counts.index, counts.values,
                color=sns.color_palette("RdYlGn", len(counts)))
    axes[0].set_xticks(range(len(names) + 1))
    axes[0].set_xlabel("# models that answered correctly")
    axes[0].set_ylabel("# samples")
    axes[0].set_title("Global sample difficulty", fontweight="bold")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(int(v)), ha="center", fontsize=9)

    # Per-class stacked
    pivot = (
        merged.groupby(["true_label", "n_correct"]).size().unstack(fill_value=0).reindex(CLASSES)
    )
    pivot = pivot.reindex(columns=range(len(names) + 1), fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)
    bottom = np.zeros(len(pivot_pct))
    colors = sns.color_palette("RdYlGn", len(names) + 1)
    for i, col in enumerate(pivot_pct.columns):
        axes[1].bar(pivot_pct.index, pivot_pct[col].values, bottom=bottom,
                    label=f"{col}/{len(names)}", color=colors[i])
        bottom += pivot_pct[col].values
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Fraction of samples")
    axes[1].set_title("Per-class model agreement with ground truth", fontweight="bold")
    axes[1].legend(title="# correct", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return merged


def main():
    dfs = load_predictions()

    summary = overall_metrics(dfs)
    summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    print("\n=== OVERALL METRICS ===")
    print(summary.to_string(index=False))

    pc_df = per_class_f1(dfs)
    pc_df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)

    kappa, merged = agreement_matrix(dfs)
    kappa.to_csv(OUT_DIR / "pairwise_kappa.csv")
    print("\n=== PAIRWISE COHEN'S KAPPA ===")
    print(kappa.astype(float).round(3))

    consensus, majority_preds, all_agree = consensus_analysis(merged)
    with open(OUT_DIR / "consensus_stats_3model.json", "w") as f:
        json.dump(consensus, f, indent=2)
    print("\n=== 3-MODEL CONSENSUS (incl. Llama, for completeness) ===")
    print(json.dumps(consensus, indent=2))

    two, agree_df, disagree_df = two_model_consensus(dfs)
    with open(OUT_DIR / "consensus_stats_2model.json", "w") as f:
        json.dump(two, f, indent=2)
    print("\n=== 2-MODEL CONSENSUS (Gemini ∩ GPT) — headline result ===")
    print(json.dumps(two, indent=2))

    plot_summary_figure(summary, pc_df, dfs, kappa, OUT_DIR / "summary_figure.png")
    plot_error_agreement(dfs, merged, OUT_DIR / "top_confusions_best_model.png")
    difficulty_df = plot_difficulty_histogram(dfs, OUT_DIR / "sample_difficulty.png")
    difficulty_df.to_csv(OUT_DIR / "per_sample_difficulty.csv", index=False)

    # Per-class recall table across models
    recall_pivot = pc_df.pivot(index="class", columns="model", values="recall").reindex(CLASSES)
    precision_pivot = pc_df.pivot(index="class", columns="model", values="precision").reindex(CLASSES)
    f1_pivot = pc_df.pivot(index="class", columns="model", values="f1").reindex(CLASSES)
    recall_pivot.to_csv(OUT_DIR / "recall_by_class.csv")
    precision_pivot.to_csv(OUT_DIR / "precision_by_class.csv")
    f1_pivot.to_csv(OUT_DIR / "f1_by_class.csv")

    print(f"\nAll analysis artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
