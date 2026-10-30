"""Demo 5: zero-shot LLM title/abstract screening for systematic reviews.

For every record the model reads the review question, the inclusion/exclusion criteria, the title and the abstract,
and P(include) is read from the logits of the first answer token ("Yes" vs "No"); no text is generated.
The criteria below are written from each published review's question (population, exposure/intervention, design).
Data: ASReview / SYNERGY open screening datasets with the reviewers' decisions (label_included = final inclusion,
label_abstract_screening = passed the human title/abstract screen, where available).
Output: preds/<dataset>.csv with record_id, score = logit(Yes) - logit(No), p_include = sigmoid(score) and the human labels.
Usage (one GPU): python screen.py [--limit 64]
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_WORDS = 350
TOKEN_BUDGET = 24000  # padded tokens per batch

REVIEWS = {
    "Appenzeller-Herzog_2020": {
        "question": "Comparative effectiveness of common therapies for Wilson disease (controlled studies).",
        "include": "Studies in patients with Wilson disease that compare a treatment (D-penicillamine, trientine, zinc salts, "
                   "tetrathiomolybdate, liver transplantation or other) with another treatment, placebo or no treatment, i.e. "
                   "studies with a control or comparison group, reporting clinical outcomes.",
        "exclude": "Single-arm case series and case reports, reviews, editorials, animal or in-vitro studies, and studies of "
                   "diagnosis, genetics or pathophysiology without a treatment comparison.",
    },
    "Wolters_2018": {
        "question": "Coronary heart disease, heart failure, and the risk of dementia.",
        "include": "Longitudinal studies (prospective or retrospective cohort, nested case-control) in community or "
                   "population-based samples of adults, in which coronary heart disease (myocardial infarction, angina, "
                   "coronary artery disease) or heart failure is the exposure and incident dementia (all-cause, Alzheimer's "
                   "disease or vascular dementia) is the outcome.",
        "exclude": "Cross-sectional studies, studies with only cognitive test scores or cognitive decline and no dementia "
                   "diagnosis, reviews, editorials, case reports, and animal studies.",
    },
    "van_Dis_2020": {
        "question": "Long-term outcomes of cognitive behavioral therapy for anxiety-related disorders.",
        "include": "Randomized controlled trials in adults with an anxiety-related disorder (generalized anxiety disorder, "
                   "panic disorder, social anxiety disorder, specific phobia, post-traumatic stress disorder or "
                   "obsessive-compulsive disorder) comparing cognitive behavioral therapy, including exposure therapy, with a "
                   "control condition (waiting list, usual care, placebo or psychological placebo).",
        "exclude": "Non-randomized studies, trials without a control condition, trials only in children or adolescents, "
                   "reviews, meta-analyses and study protocols.",
    },
}
SYSTEM = ("You screen titles and abstracts for a systematic review. Answer with one word: Yes if the record should go to "
          "full-text review, No otherwise. If the abstract is missing or does not give enough detail, answer Yes.")


def prompt(review, title, abstract):
    words = str(abstract).split() if isinstance(abstract, str) and abstract.strip() else []
    abstract = " ".join(words[:MAX_WORDS]) + (" ..." if len(words) > MAX_WORDS else "") if words else "not available"
    return (f"Review question: {review['question']}\nInclude: {review['include']}\nExclude: {review['exclude']}\n\n"
            f"Title: {title if isinstance(title, str) else 'not available'}\nAbstract: {abstract}\n\n"
            "Should this record go to full-text review? Answer Yes or No.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N records per dataset, for a quick test")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_FRACTION", "0.3")), 0)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    os.makedirs(os.path.join(ROOT, "preds"), exist_ok=True)
    log = {}
    for name, review in REVIEWS.items():
        df = pd.read_csv(os.path.join(ROOT, "data", f"{name}.csv"))
        if a.limit:
            df = df.head(a.limit)
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": prompt(review, r.title, r.abstract)}],
                                         tokenize=False, add_generation_prompt=True) for r in df.itertuples()]
        lens = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: lens[i])
        score = [0.0] * len(texts)  # logit(Yes) - logit(No): keeps the ranking where P(include) rounds to 0 or 1
        t0, i = time.time(), 0
        while i < len(order):
            bs = 1  # lengths are sorted, so the last record in the batch is the longest
            while i + bs < len(order) and bs < 64 and (bs + 1) * lens[order[i + bs]] <= TOKEN_BUDGET:
                bs += 1
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, [yes, no]].float()
            for j, v in zip(idx, (logits[:, 0] - logits[:, 1]).tolist()):
                score[j] = v
            i += bs
        sec = time.time() - t0
        score_t = torch.tensor(score)
        out = pd.DataFrame({"record_id": df.record_id, "score": score, "p_include": torch.sigmoid(score_t).tolist(),
                            "label_included": df.label_included,
                            "has_abstract": df.abstract.notna()})
        if "label_abstract_screening" in df:
            out["label_abstract_screening"] = df.label_abstract_screening
        out.to_csv(os.path.join(ROOT, "preds", f"{name}.csv"), index=False)
        log[name] = {"records": len(df), "seconds": round(sec, 1), "prompt_tokens": int(sum(lens)),
                     "tokens_per_s": round(sum(lens) / sec), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(name, json.dumps(log[name]), flush=True)
    log["model"] = MODEL
    log["gpu"] = torch.cuda.get_device_name(0)
    json.dump(log, open(os.path.join(ROOT, "preds", "run_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
