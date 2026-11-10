import json

import httpx
import numpy as np
import pytest
import respx

from quantgap.backends import BackendSettings, ResolvedSource
from quantgap.backends.openai_compat import OpenAIBackend
from quantgap.score.run import score_quant
from quantgap.tokcheck import CHECK_STRING

BASE = "http://fake:8000/v1"


def _fake_server(model, top_n=5, max_n=20):
    import torch

    def completions(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        ids = body["prompt"]
        if body.get("prompt_logprobs") is not None:
            n = body["prompt_logprobs"]
            if n > max_n:
                return httpx.Response(400, json={"error": f"prompt_logprobs must be <= {max_n}"})
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
            entries = [None]
            for i in range(1, len(ids)):
                row = lp[i - 1]
                top = torch.topk(row, min(n, top_n))
                e = {
                    str(int(t)): {"logprob": float(v), "rank": r + 1}
                    for r, (v, t) in enumerate(zip(top.values, top.indices, strict=True))
                }
                e.setdefault(str(ids[i]), {"logprob": float(row[ids[i]]), "rank": 99})
                entries.append(e)
            return httpx.Response(200, json={"choices": [{"prompt_logprobs": entries, "text": ""}]})
        inp = torch.tensor([ids])
        out = model.generate(
            input_ids=inp,
            attention_mask=torch.ones_like(inp),
            max_new_tokens=body["max_tokens"],
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=2,
        )
        new = out[0, len(ids) :].tolist()
        toks = [f"token_id:{t}" for t in new]
        return httpx.Response(200, json={"choices": [{"logprobs": {"tokens": toks}}]})

    return completions


def _mount(router, model, tok, **kw):
    router.post(f"{BASE}/completions").mock(side_effect=_fake_server(model, **kw))
    ids = tok(CHECK_STRING, add_special_tokens=False)["input_ids"]
    router.post("http://fake:8000/tokenize").mock(
        side_effect=lambda r: httpx.Response(200, json={"tokens": ids})
    )
    router.post("http://fake:8000/detokenize").mock(
        side_effect=lambda r: httpx.Response(
            200,
            json={"prompt": tok.decode(json.loads(r.content)["tokens"], skip_special_tokens=True)},
        )
    )


def _backend(n=20):
    b = OpenAIBackend()
    s = BackendSettings(base_url=BASE, model_name="tiny", ctx=2048, max_prompt_logprobs=n)
    src = ResolvedSource(
        source="openai://tiny", label="tiny-server", format="openai", extra={"model_name": "tiny"}
    )
    b.load(src, s)
    return b, s, src


@respx.mock
def test_t9_top5_server_is_approximate_and_still_sane(tiny_hf, tiny_pack):
    model, tok = tiny_hf
    _mount(respx, model, tok, top_n=5)
    b, s, src = _backend()
    res = score_quant(tiny_pack, b, src, s, do_damage=False)
    assert res.approximate is True
    assert any("approximate" in w for w in res.warnings)
    assert np.isfinite(res.metrics.kld_mean) and res.metrics.kld_mean >= 0
    assert res.metrics.top1_agree == pytest.approx(1.0)
    assert res.metrics.ppl_ratio == pytest.approx(1.0, abs=1e-4)


@respx.mock
def test_server_that_returns_everything_is_not_approximate(tiny_hf, tiny_pack):
    model, tok = tiny_hf
    _mount(respx, model, tok, top_n=512, max_n=512)
    b, s, src = _backend(n=512)
    res = score_quant(tiny_pack, b, src, s, do_damage=False)
    assert not res.approximate
    assert res.metrics.kld_mean < 1e-6


@respx.mock
def test_n_steps_down_when_server_refuses(tiny_hf):
    model, tok = tiny_hf
    _mount(respx, model, tok, top_n=5, max_n=10)
    b, _, _ = _backend(n=50)
    b._request_prompt_logprobs([1, 5, 6, 7])
    assert b.n <= 10


@respx.mock
def test_damage_checks_over_http(tiny_hf, tiny_pack):
    model, tok = tiny_hf
    _mount(respx, model, tok)
    b, s, src = _backend()
    res = score_quant(tiny_pack, b, src, s, do_damage=True, quick=True)
    assert res.metrics.damage is not None and res.metrics.damage.n_prompts > 0
