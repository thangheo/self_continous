"""Chạy thí nghiệm. Kết quả ghi ra results/ dưới dạng jsonl + npz.

Design:
  text  x  condition {A,B,C}  x  mode {chained,direct}  x  seed  x  iteration

  chained : R_i = rewrite(R_{i-1})     -> drift có tích luỹ hay không
  direct  : R_i = rewrite(R_0)         -> control, mỗi điểm là 1 bước độc lập
            từ cùng gốc. Nếu chained ≈ direct thì không có compounding,
            chỉ là single-step noise.
"""
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from config import CFG
from embed import get_embedder
import llm


def load_texts(d):
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".txt"):
            out.append((fn[:-4], open(os.path.join(d, fn), encoding="utf-8").read().strip()))
    if not out:
        sys.exit(f"Không có .txt nào trong {d}/")
    return out


def generate(cfg, cache, texts):
    """Sinh toàn bộ versions -> results/versions.jsonl"""
    path = os.path.join(cfg.out_dir, "versions.jsonl")
    rows = []
    total = len(texts) * len(cfg.conditions) * len(cfg.modes) * cfg.n_seeds * cfg.n_iterations
    bar = tqdm(total=total, desc="rewrites")

    for tid, original in texts:
        for cond in cfg.conditions:
            for mode in cfg.modes:
                for seed in range(cfg.n_seeds):
                    rows.append(dict(text_id=tid, condition=cond, mode=mode,
                                     seed=seed, iteration=0, text=original))
                    cur = original
                    for it in range(1, cfg.n_iterations + 1):
                        src = cur if mode == "chained" else original
                        cur = llm.rewrite(cache, cfg, src, cond,
                                          nonce=(tid, cond, mode, seed, it))
                        rows.append(dict(text_id=tid, condition=cond, mode=mode,
                                         seed=seed, iteration=it, text=cur))
                        bar.update(1)
    bar.close()

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {path}  ({len(rows)} versions)")
    return rows


def embed_all(cfg, rows):
    emb = get_embedder(cfg)
    vecs = emb([r["text"] for r in tqdm(rows, desc="embed")])
    np.savez_compressed(os.path.join(cfg.out_dir, "embeddings.npz"), Z=vecs)
    print(f"-> embeddings.npz  {vecs.shape}")
    return vecs


def judge_all(cfg, cache, texts, rows):
    """Claim preservation. Chỉ chấm ở cfg.judge_iterations cho đỡ tốn."""
    claim_sets = {}
    for tid, original in tqdm(texts, desc="extract claims"):
        claim_sets[tid] = llm.extract_claims(cache, cfg, original)
    json.dump(claim_sets, open(os.path.join(cfg.out_dir, "claims.json"), "w"),
              ensure_ascii=False, indent=2)

    todo = [r for r in rows if r["iteration"] in cfg.judge_iterations]
    out = []
    for r in tqdm(todo, desc="judge"):
        claims = claim_sets[r["text_id"]]
        votes = [llm.judge_claims(cache, cfg, claims, r["text"], nonce=v)
                 for v in range(cfg.judge_votes)]
        labels = _majority([v["labels"] for v in votes])
        out.append({k: r[k] for k in ("text_id", "condition", "mode", "seed", "iteration")}
                   | {"labels": labels,
                      "n_invented": int(np.mean([len(v["invented"]) for v in votes])),
                      "agreement": _agreement([v["labels"] for v in votes])})

    p = os.path.join(cfg.out_dir, "judgements.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {p}  ({len(out)} judged)")


def _majority(votelists):
    if len(votelists) == 1:
        return votelists[0]
    return [max(set(col), key=col.count) for col in zip(*votelists)]


def _agreement(votelists):
    """Tỉ lệ claim mà mọi vote đồng ý. <0.8 => judge không đủ tin cậy."""
    if len(votelists) == 1:
        return 1.0
    return float(np.mean([len(set(col)) == 1 for col in zip(*votelists)]))


if __name__ == "__main__":
    cfg = CFG
    os.makedirs(cfg.out_dir, exist_ok=True)
    cache = llm.Cache(cfg.cache_path)
    texts = load_texts(cfg.text_dir)
    print(f"{len(texts)} texts, "
          f"{len(texts)*len(cfg.conditions)*len(cfg.modes)*cfg.n_seeds} trajectories")

    rows = generate(cfg, cache, texts)
    embed_all(cfg, rows)
    if "--no-judge" not in sys.argv:
        judge_all(cfg, cache, texts, rows)
    print("Xong. Chạy: python analyze.py")
