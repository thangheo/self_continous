"""Anthropic wrapper + prompts + sqlite cache.

Cache là bắt buộc: một lần chạy full design là vài nghìn call, crash giữa chừng
mà không có cache thì đốt tiền lần hai.
"""
import hashlib
import json
import os
import sqlite3
import time
from typing import List, Dict, Any

import anthropic

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()   # đọc ANTHROPIC_API_KEY từ env
    return _client


# ---------------------------------------------------------------- cache
class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)"
        )
        self.db.commit()

    def get(self, k: str):
        row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))
        self.db.commit()


def _key(**kw) -> str:
    return hashlib.sha256(
        json.dumps(kw, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ---------------------------------------------------------------- call
def complete(cache: Cache, model: str, system: str, user: str,
             max_tokens: int, nonce: Any = 0) -> str:
    """nonce phân biệt các sample độc lập cùng prompt (API không có seed param)."""
    k = _key(model=model, system=system, user=user, mt=max_tokens, nonce=nonce)
    hit = cache.get(k)
    if hit is not None:
        return hit

    for attempt in range(6):
        try:
            r = client().messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out = "".join(b.text for b in r.content if b.type == "text").strip()
            cache.put(k, out)
            return out
        except Exception as e:                      # rate limit / overload
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- prompts
REWRITE_SYSTEM = (
    "You are a text rewriting engine. Output ONLY the rewritten passage. "
    "No preamble, no commentary, no markdown fences, no title."
)

REWRITE_INSTRUCTIONS = {
    # A: cách người ta thực sự dùng LLM để "làm cho hay hơn"
    "A_natural":
        "Rewrite the passage below to make it clearer and better written. "
        "Keep roughly the same length.",
    # B: có ràng buộc bảo toàn nội dung
    "B_preserve":
        "Rewrite the passage below for clarity. You MUST preserve every factual "
        "claim, every quantity, and every logical relation between claims. "
        "Do not add, remove, merge, or soften any proposition. "
        "Keep roughly the same length.",
    # C: baseline stylistic-only -> dùng để trừ ra phần drift do văn phong
    "C_style_only":
        "Fix grammar, punctuation and word choice in the passage below. "
        "Do NOT change sentence order, do NOT add or remove information, "
        "do NOT merge or split propositions. Minimal edits only.",
}


def rewrite(cache, cfg, text: str, condition: str, nonce) -> str:
    return complete(
        cache, cfg.rewrite_model, REWRITE_SYSTEM,
        f"{REWRITE_INSTRUCTIONS[condition]}\n\n---\n{text}\n---",
        cfg.max_tokens, nonce=nonce,
    )


# ---------------------------------------------------------------- claims
CLAIM_SYSTEM = (
    "You extract atomic propositions from text. "
    "Return ONLY a JSON array of strings. No markdown, no commentary."
)

CLAIM_USER = """Extract the atomic claims of the passage below.

Rules:
- One proposition per item, self-contained, no pronouns.
- Include factual assertions, quantities, causal and logical relations
  (e.g. "X therefore Y" is a claim about the inference, list it separately).
- 8-20 items. Order by appearance.

---
{text}
---"""


def extract_claims(cache, cfg, text: str) -> List[str]:
    raw = complete(cache, cfg.judge_model, CLAIM_SYSTEM,
                   CLAIM_USER.format(text=text), cfg.max_tokens)
    return _parse_json(raw, default=[])


JUDGE_SYSTEM = (
    "You are a strict claim-verification judge. "
    "Return ONLY JSON. No markdown, no commentary."
)

JUDGE_USER = """You are given a list of claims from an ORIGINAL passage, and a REWRITTEN passage.

For each original claim, assign exactly one label based ONLY on the rewritten passage:
- PRESERVED    : asserted with the same force and same content
- WEAKENED     : still present but hedged, vaguer, or lost a qualifier/quantity
- MERGED       : survives only fused into another claim, losing its own identity
- DROPPED      : not recoverable from the rewritten passage
- CONTRADICTED : the rewritten passage asserts something incompatible

Then list INVENTED claims: substantive propositions in the rewritten passage that
are not entailed by any original claim. Exclude pure rephrasing and connective filler.

Output JSON exactly:
{{"labels": ["PRESERVED", ...], "invented": ["...", "..."]}}
labels must have exactly {n} entries, in the same order as the claims.

ORIGINAL CLAIMS:
{claims}

REWRITTEN PASSAGE:
---
{text}
---"""


def judge_claims(cache, cfg, claims: List[str], text: str, nonce=0) -> Dict:
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    raw = complete(
        cache, cfg.judge_model, JUDGE_SYSTEM,
        JUDGE_USER.format(n=len(claims), claims=numbered, text=text),
        cfg.max_tokens, nonce=nonce,
    )
    d = _parse_json(raw, default={"labels": [], "invented": []})
    if isinstance(d, list):                         # model trả thẳng array labels
        d = {"labels": d, "invented": []}
    labels = d.get("labels", [])
    if len(labels) != len(claims):                 # pad/truncate an toàn
        labels = (labels + ["DROPPED"] * len(claims))[: len(claims)]
    return {"labels": labels, "invented": d.get("invented", [])}


def _parse_json(raw: str, default):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        s = s[4:] if s.lower().startswith("json") else s
    try:
        return json.loads(s.strip())
    except Exception:
        i, j = s.find("["), s.rfind("]")
        k, l = s.find("{"), s.rfind("}")
        for a, b in ((k, l), (i, j)):
            if a != -1 and b > a:
                try:
                    return json.loads(s[a:b + 1])
                except Exception:
                    pass
        return default
