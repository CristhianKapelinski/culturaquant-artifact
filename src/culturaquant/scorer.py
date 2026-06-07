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
from dataclasses import dataclass

import torch

LETTERS = ["A", "B", "C", "D", "E"]

PROMPT_HEADER = (
    "Responda a pergunta de multipla escolha escolhendo a unica alternativa correta.\n\n"
)


def build_prompt(question: str, options: list[str]) -> tuple[str, list[str]]:
    """Return (context, continuations). The context ends right before the answer
    letter; each continuation is one lettered answer line."""
    lines = [PROMPT_HEADER + f"Pergunta: {question}\n"]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}) {opt}")
    lines.append("\nResposta:")
    context = "\n".join(lines[:1]) + "\n" + "\n".join(lines[1:-1]) + "\n" + lines[-1]
    continuations = [f" {letter}" for letter in LETTERS[: len(options)]]
    return context, continuations


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
def score_item(lm: LoadedModel, question: str, options: list[str]) -> int:
    """Return the predicted option index by length-normalized log-likelihood."""
    context, continuations = build_prompt(question, options)
    tok = lm.tokenizer
    device = next(lm.model.parameters()).device

    ctx_ids = tok(context, return_tensors="pt", add_special_tokens=True).input_ids[0]
    best_idx = 0
    best_score = -float("inf")
    for i, cont in enumerate(continuations):
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
        norm = logprob / max(1, len(cont_ids))
        if norm > best_score:
            best_score = norm
            best_idx = i
    return best_idx


def free_model(lm: LoadedModel) -> None:
    del lm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
