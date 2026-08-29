"""Metrics + figures.

Câu hỏi thật sự cần trả lời không phải "distance có tăng không" (chắc chắn tăng),
mà là drift thuộc regime nào:

  R_n = ||z_n - z_0||  ~  n^alpha

  alpha ~ 0.5  -> random walk / diffusion thuần
  alpha ~ 1.0  -> ballistic, có hướng ưu tiên (model kéo text về prior của nó)
  alpha -> 0   -> saturation: có attractor, text hội tụ về "house style" rồi đứng im

Ba khả năng này cho ba kết luận rất khác nhau. Phân biệt chúng bằng:
  (a) exponent alpha fit trên log-log
  (b) step autocorrelation: random walk -> ~0, ballistic -> > 0
  (c) ensemble dispersion giữa các seed: diffusion -> nở ra, attractor -> co lại
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from embed import angular

OUT = CFG.out_dir
LABELS = ["PRESERVED", "WEAKENED", "MERGED", "DROPPED", "CONTRADICTED"]


def load():
    rows = [json.loads(l) for l in open(f"{OUT}/versions.jsonl", encoding="utf-8")]
    Z = np.load(f"{OUT}/embeddings.npz")["Z"]
    idx = {}
    for i, r in enumerate(rows):
        idx[(r["text_id"], r["condition"], r["mode"], r["seed"], r["iteration"])] = i
    return rows, Z, idx


def trajectories(rows, Z, idx):
    """-> dict[(text,cond,mode,seed)] = array (n_iter+1, dim)"""
    keys = {(r["text_id"], r["condition"], r["mode"], r["seed"]) for r in rows}
    out = {}
    for k in keys:
        out[k] = np.stack([Z[idx[k + (it,)]] for it in range(CFG.n_iterations + 1)])
    return out


# ---------------------------------------------------------------- metrics
def drift_curves(trajs):
    """R_n (displacement), D_n (path length), v_n (step) gộp theo cond/mode."""
    acc = defaultdict(lambda: defaultdict(list))
    for (tid, cond, mode, seed), T in trajs.items():
        R = angular(T[0][None, :], T)                 # (n+1,)
        v = angular(T[:-1], T[1:])                    # (n,)
        D = np.concatenate([[0.0], np.cumsum(v)])
        acc[(cond, mode)]["R"].append(R)
        acc[(cond, mode)]["D"].append(D)
        acc[(cond, mode)]["v"].append(np.concatenate([[0.0], v]))
        acc[(cond, mode)]["tid"].append(tid)
    return acc


def boot_ci(curves, tids, B=2000):
    """Bootstrap trên đơn vị text, không phải trên seed. Seed không độc lập về
    mặt generalization: 20 seed của 1 text vẫn chỉ là 1 text."""
    A = np.stack(curves)
    tids = np.array(tids)
    uniq = np.unique(tids)
    per_text = np.stack([A[tids == t].mean(0) for t in uniq])
    m = per_text.mean(0)
    if len(uniq) < 2:
        return m, m, m
    bs = np.stack([per_text[np.random.randint(0, len(uniq), len(uniq))].mean(0)
                   for _ in range(B)])
    return m, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def scaling_exponent(R_mean):
    """Fit log R_n = alpha*log n + c, bỏ n=0 và bỏ vùng bão hoà (R > 0.9*pi)."""
    n = np.arange(1, len(R_mean))
    r = R_mean[1:]
    m = (r > 1e-6) & (r < 0.9 * np.pi)
    if m.sum() < 3:
        return np.nan, np.nan
    a, c = np.polyfit(np.log(n[m]), np.log(r[m]), 1)
    pred = a * np.log(n[m]) + c
    ss = 1 - np.sum((np.log(r[m]) - pred) ** 2) / np.sum((np.log(r[m]) - np.log(r[m]).mean()) ** 2)
    return float(a), float(ss)


def step_autocorr(trajs, cond, mode):
    """cos giữa 2 bước dịch chuyển liên tiếp. >0 = có hướng, ~0 = random walk."""
    vals = []
    for (tid, c, m, s), T in trajs.items():
        if (c, m) != (cond, mode):
            continue
        d = np.diff(T, axis=0)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        vals += list(np.sum(d[:-1] * d[1:], axis=1))
    return float(np.mean(vals)) if vals else np.nan


def dispersion(trajs, cond, mode):
    """Khoảng cách trung bình giữa các seed tại mỗi iteration.
    Nở ra = diffusion. Co lại = attractor / mode collapse."""
    per_text = defaultdict(list)
    for (tid, c, m, s), T in trajs.items():
        if (c, m) == (cond, mode):
            per_text[tid].append(T)
    out = []
    for tid, Ts in per_text.items():
        A = np.stack(Ts)                     # (seeds, iters, dim)
        d = []
        for it in range(A.shape[1]):
            X = A[:, it]
            dm = angular(X[:, None, :], X[None, :, :])
            d.append(dm[np.triu_indices(len(X), 1)].mean() if len(X) > 1 else 0.0)
        out.append(d)
    return np.array(out).mean(0) if out else None


def claim_curves():
    p = f"{OUT}/judgements.jsonl"
    if not os.path.exists(p):
        return None
    J = [json.loads(l) for l in open(p, encoding="utf-8")]
    agg = defaultdict(lambda: defaultdict(list))
    for r in J:
        n = len(r["labels"]) or 1
        counts = {L: r["labels"].count(L) / n for L in LABELS}
        agg[(r["condition"], r["mode"])][r["iteration"]].append(
            counts | {"invented": r["n_invented"], "agree": r["agreement"]})
    return agg


# ---------------------------------------------------------------- figures
def fig_trajectories(trajs):
    try:
        import umap
        red, name = umap.UMAP(n_components=2, metric="cosine",
                              random_state=0), "UMAP"
    except Exception:
        from sklearn.decomposition import PCA
        red, name = PCA(n_components=2), "PCA"

    keys = [k for k in trajs if k[2] == "chained"]
    tids = sorted({k[0] for k in keys})
    tid = tids[0]
    keys = [k for k in keys if k[0] == tid]
    X = np.concatenate([trajs[k] for k in keys])
    Y = red.fit_transform(X)

    colors = {"A_natural": "tab:red", "B_preserve": "tab:blue", "C_style_only": "tab:green"}
    fig, ax = plt.subplots(figsize=(7, 6))
    off = 0
    seen = set()
    for k in keys:
        n = len(trajs[k])
        y = Y[off:off + n]; off += n
        c = colors.get(k[1], "gray")
        ax.plot(y[:, 0], y[:, 1], "-o", ms=3, lw=1, alpha=.6, color=c,
                label=k[1] if k[1] not in seen else None)
        seen.add(k[1])
        ax.scatter(*y[0], s=90, marker="*", color="k", zorder=5)
    ax.set_title(f"F1 · {name} semantic trajectories (chained) — {tid}\n★ = original")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/f1_trajectories.png", dpi=150)


def fig_drift(acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, ["chained", "direct"]):
        for cond in CFG.conditions:
            if (cond, mode) not in acc:
                continue
            d = acc[(cond, mode)]
            m, lo, hi = boot_ci(d["R"], d["tid"])
            x = np.arange(len(m))
            ax.plot(x, m, "-o", ms=3, label=cond)
            ax.fill_between(x, lo, hi, alpha=.18)
        ax.set_title(f"{mode}"); ax.set_xlabel("rewrite iteration"); ax.grid(alpha=.3)
    axes[0].set_ylabel("angular displacement từ original")
    axes[0].legend()
    fig.suptitle("F2 · Drift vs iteration (dải = 95% bootstrap CI trên text)")
    fig.tight_layout(); fig.savefig(f"{OUT}/f2_drift.png", dpi=150)


def fig_scaling(acc, trajs):
    fig, ax = plt.subplots(figsize=(7, 5))
    lines = []
    for cond in CFG.conditions:
        k = (cond, "chained")
        if k not in acc:
            continue
        m, _, _ = boot_ci(acc[k]["R"], acc[k]["tid"])
        a, r2 = scaling_exponent(m)
        ac = step_autocorr(trajs, cond, "chained")
        n = np.arange(1, len(m))
        ax.loglog(n, m[1:], "-o", ms=4, label=f"{cond}  α={a:.2f} (R²={r2:.2f})")
        lines.append((cond, a, r2, ac))
    n = np.arange(1, CFG.n_iterations + 1)
    ref = acc[(CFG.conditions[0], "chained")]
    m0, _, _ = boot_ci(ref["R"], ref["tid"])
    base = m0[1] if m0[1] > 0 else 1e-3
    ax.loglog(n, base * n ** 0.5, "k--", lw=1, label="α=0.5 (random walk)")
    ax.loglog(n, base * n ** 1.0, "k:", lw=1, label="α=1.0 (ballistic)")
    ax.set_xlabel("iteration (log)"); ax.set_ylabel("displacement (log)")
    ax.set_title("F3 · Scaling regime của drift"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/f3_scaling.png", dpi=150)

    print("\n=== Drift regime (chained) ===")
    for cond, a, r2, ac in lines:
        print(f"{cond:16s} alpha={a:5.2f}  R2={r2:4.2f}  step-autocorr={ac:+.3f}")


def fig_dispersion(trajs):
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in CFG.conditions:
        d = dispersion(trajs, cond, "chained")
        if d is not None:
            ax.plot(d, "-o", ms=3, label=cond)
    ax.set_xlabel("iteration"); ax.set_ylabel("khoảng cách TB giữa các seed")
    ax.set_title("F4 · Ensemble dispersion\nnở ra = diffusion · co lại = attractor")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/f4_dispersion.png", dpi=150)


def fig_claims(agg):
    if not agg:
        print("(bỏ qua F5: chưa có judgements.jsonl)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for cond in CFG.conditions:
        k = (cond, "chained")
        if k not in agg:
            continue
        its = sorted(agg[k])
        pres = [np.mean([r["PRESERVED"] for r in agg[k][i]]) for i in its]
        inv = [np.mean([r["invented"] for r in agg[k][i]]) for i in its]
        axes[0].plot(its, pres, "-o", label=cond)
        axes[1].plot(its, inv, "-o", label=cond)
    axes[0].set_ylabel("tỉ lệ claim PRESERVED"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("số claim INVENTED / version")
    for a in axes:
        a.set_xlabel("iteration"); a.grid(alpha=.3); a.legend()
    fig.suptitle("F5 · Claim preservation & invention (chained)")
    fig.tight_layout(); fig.savefig(f"{OUT}/f5_claims.png", dpi=150)

    ag = [r["agree"] for k in agg for i in agg[k] for r in agg[k][i]]
    if ag:
        print(f"\nJudge self-agreement: {np.mean(ag):.3f}  (<0.80 => kết quả claim không dùng được)")


if __name__ == "__main__":
    rows, Z, idx = load()
    trajs = trajectories(rows, Z, idx)
    acc = drift_curves(trajs)

    fig_trajectories(trajs)
    fig_drift(acc)
    fig_scaling(acc, trajs)
    fig_dispersion(trajs)
    fig_claims(claim_curves())

    # style-corrected drift: A trừ đi baseline stylistic C
    ka, kc = ("A_natural", "chained"), ("C_style_only", "chained")
    if ka in acc and kc in acc:
        a, _, _ = boot_ci(acc[ka]["R"], acc[ka]["tid"])
        c, _, _ = boot_ci(acc[kc]["R"], acc[kc]["tid"])
        print("\n=== Style-corrected drift (A - C) ===")
        for i in range(0, len(a), max(1, len(a)//5)):
            print(f"  n={i:2d}  A={a[i]:.3f}  C={c[i]:.3f}  Δ={a[i]-c[i]:+.3f}")

    print(f"\nFigures -> {OUT}/f1..f5*.png")
