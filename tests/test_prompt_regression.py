"""Regression invariant: injecting an empty/None context must not change the
prompt or the continuations. This protects the paired FP16-vs-quant scoring loop:
the 5 lettered continuations must stay byte-identical so ``score_item`` is reused
verbatim across the closed-book and RAG conditions (spec section 3 injection).
"""

from culturaquant.scorer import build_prompt

Q = "Qual e a capital do estado da Bahia?"
OPTS = ["Salvador", "Feira de Santana", "Ilheus", "Porto Seguro", "Juazeiro"]


def test_context_default_none_empty_are_byte_identical():
    base_prompt, base_cont = build_prompt(Q, OPTS)
    none_prompt, none_cont = build_prompt(Q, OPTS, context=None)
    empty_prompt, empty_cont = build_prompt(Q, OPTS, context="")
    assert none_prompt == base_prompt
    assert empty_prompt == base_prompt
    assert none_cont == base_cont == empty_cont


def test_context_injection_changes_only_the_prefix():
    base_prompt, base_cont = build_prompt(Q, OPTS)
    ctx_prompt, ctx_cont = build_prompt(Q, OPTS, context="Salvador e a capital.")
    # continuations untouched
    assert ctx_cont == base_cont
    # a Contexto: block was injected before the question
    assert "Contexto: Salvador e a capital." in ctx_prompt
    assert ctx_prompt != base_prompt
    # the question, options and the Resposta: suffix survive verbatim
    assert f"Pergunta: {Q}" in ctx_prompt
    assert ctx_prompt.endswith(base_prompt[base_prompt.index("Pergunta:"):])
    assert base_cont == [" A", " B", " C", " D", " E"]
