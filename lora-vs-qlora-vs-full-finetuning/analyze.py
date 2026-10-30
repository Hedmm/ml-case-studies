"""Demo 1 analysis: accuracy and macro-F1 with 95% bootstrap CIs, paired McNemar tests between arms.

Reads results/<arm>/{metrics.json,preds.jsonl}; writes results/summary.json.
"""
import json
import os

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = ["base", "lora", "qlora", "full"]
B = 10000


def load(arm):
    d = os.path.join(ROOT, "results", arm)
    if not os.path.exists(os.path.join(d, "metrics.json")):
        return None
    m = json.load(open(os.path.join(d, "metrics.json")))
    p = [json.loads(l) for l in open(os.path.join(d, "preds.jsonl"))]
    return m, p


def main():
    rng = np.random.default_rng(0)
    runs = {a: load(a) for a in ARMS}
    runs = {a: r for a, r in runs.items() if r}
    n = len(next(iter(runs.values()))[1])
    idx = rng.integers(0, n, size=(B, n))
    out = {"n_test": n, "bootstrap_resamples": B, "arms": {}, "mcnemar": []}
    correct = {}
    for a, (m, p) in runs.items():
        c = np.array([x["correct"] for x in p], dtype=float)
        correct[a] = c
        gold = [x["gold"] for x in p]
        pred = [x["pred"] for x in p]
        boot = c[idx].mean(axis=1)
        out["arms"][a] = {**m,
                          "accuracy_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                          "macro_f1": float(f1_score(gold, pred, average="macro", labels=sorted(set(gold)), zero_division=0))}
    names = list(runs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            only_a = int(((correct[a] == 1) & (correct[b] == 0)).sum())
            only_b = int(((correct[a] == 0) & (correct[b] == 1)).sum())
            pval = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            diff = correct[a][idx].mean(axis=1) - correct[b][idx].mean(axis=1)
            out["mcnemar"].append({"a": a, "b": b, "only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(pval),
                                   "acc_diff": float(correct[a].mean() - correct[b].mean()),
                                   "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]})
    json.dump(out, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
    for a, r in out["arms"].items():
        print(f"{a:6s} acc {r['accuracy']:.4f} CI [{r['accuracy_ci95'][0]:.4f},{r['accuracy_ci95'][1]:.4f}] "
              f"F1 {r['macro_f1']:.4f} valid {r['valid_label_rate']:.4f} train {r['train_seconds']}s mem {r['peak_gpu_mem_gb']}GB "
              f"trainable {r['trainable_params']:,}")
    for t in out["mcnemar"]:
        print(f"{t['a']} vs {t['b']}: diff {t['acc_diff']:+.4f} CI [{t['diff_ci95'][0]:+.4f},{t['diff_ci95'][1]:+.4f}] p={t['p_value']:.2e}")


if __name__ == "__main__":
    main()
