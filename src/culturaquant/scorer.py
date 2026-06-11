"""Constrained log-likelihood MCQ scoring (lm-evaluation-harness style).

For each item we build a fixed PT-BR prompt listing the 5 lettered options and
score each option by the length-normalized log-likelihood of its continuation
under the model. The predicted answer is the highest-likelihood option. This is
deterministic (no sampling) and identical across precisions, so a per-item
FP16-vs-quantized comparison is a clean paired measurement.

Models are loaded at a chosen precision: fp16 (reference), bnb-int8, or bnb-nf4
(4-bit). The SAME prompts and SAME items are scored at every precision.
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass

import torch

LETTERS = ["A", "B", "C", "D", "E"]


def option_permutation(item_id: str, n_options: int, seed: int) -> list[int]:
    """Deterministic per-item permutation of option indices.

    The author-written cultural and control items ship with the correct answer
    fixed at index 0 (canonical order). Scoring them in that order makes the gold
    answer always "A", so any positional bias toward the first option inflates
    accuracy; NF4 quantization in particular collapses toward "A" and scores
    spuriously high. We therefore shuffle the 5 options with a permutation that is
    a deterministic function of (``item_id``, ``seed``), so the gold position is
    balanced across the item set and is reproducible. The SAME permutation must be
    applied to every precision and to the RAG pass for one item, so the paired
    cross-precision QDIS comparison stays clean.

    Returns ``perm`` such that ``new_options[i] = old_options[perm[i]]``.
    """
    h = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    rank = sorted(range(n_options), key=lambda i: h[i % len(h)] * n_options + i)
    return rank


def apply_permutation(
    options: list[str], gold_index: int, perm: list[int]
) -> tuple[list[str], int]:
    """Reorder ``options`` by ``perm`` and return (new_options, new_gold_index)."""
    new_options = [options[p] for p in perm]
    new_gold = perm.index(gold_index)
    return new_options, new_gold

PROMPT_HEADER = (
    "Responda a pergunta de multipla escolha escolhendo a unica alternativa correta.\n\n"
)


def build_prompt(
    question: str, options: list[str], context: str | None = None
) -> tuple[str, list[str]]:
    """Return (prompt, continuations). The prompt ends right before the answer
    letter; each continuation is one lettered answer line.

    When ``context`` is a non-empty string a ``Contexto:`` block is injected
    BEFORE the question. The option lines and the trailing ``\\nResposta:`` suffix
    stay byte-identical, so the 5 lettered continuations are unchanged and
    ``score_item`` is reused verbatim across the closed-book and RAG conditions.

    Regression invariant: ``build_prompt(q, o)``, ``build_prompt(q, o, None)`` and
    ``build_prompt(q, o, '')`` all return byte-identical output to the no-context
    form (an empty/None context injects nothing).
    """
    header = PROMPT_HEADER
    if context:
        header = PROMPT_HEADER + f"Contexto: {context}\n\n"
    lines = [header + f"Pergunta: {question}\n"]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}) {opt}")
    lines.append("\nResposta:")
    prompt = "\n".join(lines[:1]) + "\n" + "\n".join(lines[1:-1]) + "\n" + lines[-1]
    continuations = [f" {letter}" for letter in LETTERS[: len(options)]]
    return prompt, continuations


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    precision: str
    name: str


def load_model(name: str, precision: str, dtype: str = "float16") -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    kwargs = {"torch_dtype": torch_dtype, "device_map": "cuda:0"}

    if precision == "fp16":
        pass
    elif precision == "int8":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
    elif precision == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)
    else:
        raise ValueError(f"unknown precision {precision}")

    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return LoadedModel(model=model, tokenizer=tok, precision=precision, name=name)


@torch.no_grad()
def score_item_logprobs(
    lm: LoadedModel,
    question: str,
    options: list[str],
    context: str | None = None,
) -> list[float]:
    """Return the per-option length-normalized continuation log-probs (the ``norm``
    vector), one float per option. This is the on-device gate signal: QDIS, the
    softmax posterior, and the argmax all derive from this vector.

    ``context`` is forwarded to ``build_prompt`` so the same scoring loop serves
    both closed-book (``context=None``) and RAG-injected (``context=passage``)
    conditions with byte-identical continuations.
    """
    prompt, continuations = build_prompt(question, options, context)
    tok = lm.tokenizer
    device = next(lm.model.parameters()).device

    ctx_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
    norms: list[float] = []
    for cont in continuations:
        cont_ids = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_ids = torch.cat([ctx_ids, cont_ids]).unsqueeze(0).to(device)
        logits = lm.model(input_ids).logits[0]
        # log-prob of each continuation token given the preceding tokens
        n_ctx = len(ctx_ids)
        logprob = 0.0
        for j, tid in enumerate(cont_ids):
            pos = n_ctx + j - 1
            lp = torch.log_softmax(logits[pos].float(), dim=-1)[tid].item()
            logprob += lp
        norms.append(logprob / max(1, len(cont_ids)))
    return norms


def score_item(lm: LoadedModel, question: str, options: list[str]) -> int:
    """Return the predicted option index by length-normalized log-likelihood.

    Thin argmax wrapper over :func:`score_item_logprobs`; behavior is unchanged
    from the original closed-book scorer.
    """
    norms = score_item_logprobs(lm, question, options)
    best_idx = 0
    best_score = -float("inf")
    for i, norm in enumerate(norms):
        if norm > best_score:
            best_score = norm
            best_idx = i
    return best_idx


def free_model(lm: LoadedModel) -> None:
    del lm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
