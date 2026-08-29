"""Tương quan độ dài <-> drift, TRONG từng condition, đã khử hiệu ứng
(text, iteration) bằng cách trừ trung bình theo ô. Gộp cả 3 condition
như lencheck.py là sai: nó đo khác biệt giữa các condition, không phải
quan hệ độ dài-drift."""
import json, collections
import numpy as np

V = [json.loads(l) for l in open('results/versions.jsonl')]
Z = np.load('results/embeddings.npz')['Z']
idx = {(r['text_id'], r['condition'], r['mode'], r['seed'], r['iteration']): i
       for i, r in enumerate(V)}
wc = {k: len(V[i]['text'].split()) for k, i in idx.items()}

for cond in ['A_natural', 'B_preserve', 'C_style_only']:
    cells = collections.defaultdict(list)
    for k, i in idx.items():
        tid, c, mode, seed, it = k
        if c != cond or mode != 'chained' or it == 0:
            continue
        base = wc[(tid, c, mode, seed, 0)]
        j = idx[(tid, c, mode, seed, 0)]
        if not base:
            continue
        L = wc[k] / base
        D = float(np.arccos(np.clip(np.dot(Z[i], Z[j]), -1, 1)))
        cells[(tid, it)].append((L, D))

    rl, rd = [], []
    for pts in cells.values():
        if len(pts) < 2:
            continue
        L = np.array([p[0] for p in pts]); D = np.array([p[1] for p in pts])
        rl += list(L - L.mean()); rd += list(D - D.mean())   # residual hoá
    if len(rl) < 8 or np.std(rl) < 1e-9:
        print(f"{cond:15s} không đủ biến thiên độ dài để kiểm (n={len(rl)})")
        continue
    r = np.corrcoef(rl, rd)[0, 1]
    print(f"{cond:15s} r_partial = {r:+.3f}   n={len(rl)}   "
          f"biến thiên độ dài trong ô: sd={np.std(rl):.3f}")

print("\nDiễn giải: đây là tương quan SAU khi khử text và iteration.")
print("  |r| > 0.5 -> trong cùng điều kiện, nén nhiều thì drift xa. Confound thật.")
print("  |r| < 0.3 -> độ dài không giải thích drift ở mức trajectory.")