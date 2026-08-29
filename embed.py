"""Embedding backends. Trả về vector đã L2-normalize.

Lưu ý phương pháp: mọi sentence embedding đều trộn lẫn style và content.
Đó là lý do phải có condition C làm baseline: drift_A - drift_C mới xấp xỉ
phần semantic. Nếu chỉ nhìn drift_A một mình thì không kết luận được gì.
"""
import numpy as np


def get_embedder(cfg):
    b = cfg.embed_backend

    if b == "st":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(cfg.embed_model)
        prefix = "query: " if "e5" in cfg.embed_model else ""

        def f(texts):
            v = m.encode([prefix + t for t in texts],
                         batch_size=16, show_progress_bar=False)
            return _norm(np.asarray(v, dtype=np.float32))
        return f

    if b == "voyage":
        import voyageai
        vo = voyageai.Client()

        def f(texts):
            out = []
            for i in range(0, len(texts), 64):
                out += vo.embed(texts[i:i + 64], model=cfg.embed_model,
                                input_type="document").embeddings
            return _norm(np.asarray(out, dtype=np.float32))
        return f

    if b == "openai":
        from openai import OpenAI
        cl = OpenAI()

        def f(texts):
            out = []
            for i in range(0, len(texts), 64):
                r = cl.embeddings.create(model=cfg.embed_model,
                                         input=texts[i:i + 64])
                out += [d.embedding for d in r.data]
            return _norm(np.asarray(out, dtype=np.float32))
        return f

    raise ValueError(f"unknown embed_backend {b}")


def _norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def angular(a, b):
    """Geodesic distance trên unit sphere. Dùng cái này thay cosine distance
    khi fit scaling law: cosine distance bão hoà ở 2, angular ở pi, và angular
    cộng tính tốt hơn dọc theo một quỹ đạo."""
    c = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(c)
