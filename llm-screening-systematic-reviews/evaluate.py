"""Demo 5 evaluation of the LLM screening scores against the reviewers' decisions.

Ranking (no threshold): ROC AUC, WSS@95 (work saved over random screening at 95% recall), recall-vs-screened curve.
Decision (with a threshold): the threshold is chosen on the development review only (Appenzeller-Herzog_2020: the
highest score that keeps >= 95% recall of its final inclusions) and then applied unchanged to the other two reviews.
At that threshold: recall of final inclusions (Wilson 95% CI), share of records the reviewer no longer has to read,
precision, and Cohen's kappa against the human title/abstract screening decisions where available.
Output: results/metrics.json, results/curves.json, results/missed.csv
Usage: python evaluate.py
"""
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DEV = "Appenzeller-Herzog_2020"
TEST = ["Wolters_2018", "van_Dis_2020"]
TARGET_RECALL = 0.95


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def ranking(df):
    d = df.sort_values("score", ascending=False)
    y = d.label_included.values
    rec = np.cumsum(y) / y.sum()
    n = len(y)
    k95 = int(np.argmax(rec >= TARGET_RECALL)) + 1
    k100 = int(np.argmax(rec >= 1.0)) + 1
    pts = sorted(set(np.linspace(0, n - 1, 200).astype(int).tolist() + [k95 - 1, k100 - 1]))
    return {"auc": round(float(roc_auc_score(y, d.score.values)), 4),
            "wss95": round((n - k95) / n - (1 - TARGET_RECALL), 4),
            "screened_for_95": k95, "screened_for_100": k100,
            "curve": [[round((i + 1) / n, 4), round(float(rec[i]), 4)] for i in pts]}


def decision(df, t):
    flag = df.score >= t
    y = df.label_included == 1
    tp = int((flag & y).sum())
    r = {"threshold": round(float(t), 3), "records": len(df), "included_final": int(y.sum()), "found": tp,
         "recall": round(tp / y.sum(), 4), "recall_ci95": wilson(tp, int(y.sum())),
         "flagged": int(flag.sum()), "workload_saved": round(1 - flag.mean(), 4),
         "precision": round(tp / max(1, flag.sum()), 4)}
    if "label_abstract_screening" in df:
        h = df.label_abstract_screening == 1
        r["human_ta_included"] = int(h.sum())
        r["kappa_vs_human_ta"] = round(float(cohen_kappa_score(h, flag)), 3)
        r["recall_of_human_ta"] = round(float((flag & h).sum() / h.sum()), 4)
        r["agreement_vs_human_ta"] = round(float((flag == h).mean()), 4)
    return r


def main():
    preds = {n: pd.read_csv(os.path.join(ROOT, "preds", f"{n}.csv")) for n in [DEV] + TEST}
    dev = preds[DEV].sort_values("score", ascending=False)
    y = dev.label_included.values
    k = int(np.argmax(np.cumsum(y) / y.sum() >= TARGET_RECALL))
    t = float(dev.score.values[k])  # highest threshold that still reaches the target recall on the dev review
    metrics, curves, missed = {"dev": DEV, "test": TEST, "threshold_from_dev": round(t, 3)}, {}, []
    for name, df in preds.items():
        rk = ranking(df)
        curves[name] = rk.pop("curve")
        metrics[name] = {"split": "dev" if name == DEV else "test", "ranking": rk, "at_dev_threshold": decision(df, t),
                         "at_p_0.5": decision(df, 0.0)}
        data = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"))
        m = df[(df.label_included == 1) & (df.score < t)].merge(data[["record_id", "title", "year"]], on="record_id")
        for r in m.itertuples():
            missed.append({"dataset": name, "record_id": r.record_id, "score": round(r.score, 2), "has_abstract": r.has_abstract,
                           "year": r.year, "title": r.title})
    tot = {k: sum(metrics[n]["at_dev_threshold"][k] for n in TEST) for k in ("records", "included_final", "found", "flagged")}
    tot["recall"] = round(tot["found"] / tot["included_final"], 4)
    tot["recall_ci95"] = wilson(tot["found"], tot["included_final"])
    tot["workload_saved"] = round(1 - tot["flagged"] / tot["records"], 4)
    metrics["test_total"] = tot
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(metrics, open(os.path.join(ROOT, "results", "metrics.json"), "w"), indent=1)
    json.dump(curves, open(os.path.join(ROOT, "results", "curves.json"), "w"))
    pd.DataFrame(missed).to_csv(os.path.join(ROOT, "results", "missed.csv"), index=False)
    print(json.dumps(metrics, indent=1))
    print(pd.DataFrame(missed).to_string())


if __name__ == "__main__":
    main()
