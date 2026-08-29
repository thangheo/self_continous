"""Cấu hình thí nghiệm. Sửa ở đây, đừng sửa rải rác trong code."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # --- models -------------------------------------------------------
    rewrite_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-5"
    max_tokens: int = 2000

    # --- embedding ----------------------------------------------------
    # "st" = sentence-transformers (local), "voyage", "openai"
    embed_backend: str = "st"
    embed_model: str = "intfloat/e5-large-v2"

    # --- design -------------------------------------------------------
    n_iterations: int = 6
    n_seeds: int = 3
    conditions: Tuple[str, ...] = ("A_natural", "B_preserve", "C_style_only")
    modes: Tuple[str, ...] = ("chained", "direct")

    # judge tốn tiền -> chỉ chấm claim ở các iteration này
    judge_iterations: Tuple[int, ...] = (0, 2, 4, 6)
    judge_votes: int = 1          # >1 = majority vote, dùng để đo judge reliability

    # --- io -----------------------------------------------------------
    text_dir: str = "texts"
    out_dir: str = "results"
    cache_path: str = "results/llm_cache.sqlite"


CFG = Config()
