"""Run the reference model once and save its top-k next-token guesses (needs `quantgap[pack]`)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from quantgap import __version__
from quantgap.models import GreedyInfo, Manifest, ModelRef, RunnerInfo, TokenizerInfo
from quantgap.pack.reader import SeqTopK
from quantgap.pack.writer import PackWriter
from quantgap.probes.registry import load_probes, probes_sha256
from quantgap.probes.render import RenderedGreedy, RenderedSeq, render_probes
from quantgap.tokcheck import CHECK_STRING

POS_CHUNK = 128  # positions per log-softmax call, keeps memory flat for 150k-token vocabularies


def tokenizer_info(tok: Any) -> TokenizerInfo:
    try:
        blob = tok.backend_tokenizer.to_str().encode("utf-8")
        digest: str | None = hashlib.sha256(blob).hexdigest()
    except Exception:
        digest = None
    vocab = len(tok)
    bos_added = bool(tok("a", add_special_tokens=True)["input_ids"][:1] == [tok.bos_token_id])
    return TokenizerInfo(
        sha256=digest,
        vocab_size=int(vocab),
        bos=bos_added and tok.bos_token_id is not None,
        bos_id=tok.bos_token_id,
        eos_id=tok.eos_token_id,
        check_ids=list(tok(CHECK_STRING, add_special_tokens=False)["input_ids"]),
    )


def chat_template_hash(tok: Any) -> str | None:
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, str):
        return hashlib.sha256(tpl.encode("utf-8")).hexdigest()
    return None


def _topk_for_sequence(model: Any, seq: RenderedSeq, k: int, device: Any) -> SeqTopK:
    """Forward pass for one sequence. Used for the unbatched path and for tests."""
    return _topk_batch(model, [seq], k, device)[0]


def _topk_batch(model: Any, seqs: list[RenderedSeq], k: int, device: Any) -> list[SeqTopK]:
    import torch

    max_len = max(len(s.token_ids) for s in seqs)
    ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
    att = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s.token_ids)] = torch.tensor(s.token_ids)
        att[i, : len(s.token_ids)] = 1
    with torch.no_grad():
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
    out: list[SeqTopK] = []
    for i, s in enumerate(seqs):
        mask = np.array(s.score_mask[1:], bool)  # position p scored if token p+1 is scored
        pos = np.nonzero(mask)[0]
        target = np.array(s.token_ids, np.int64)[pos + 1]
        tops_i, tops_v, tgt_lp = [], [], []
        for a in range(0, len(pos), POS_CHUNK):
            sel = torch.from_numpy(pos[a : a + POS_CHUNK]).to(device)
            lp = torch.log_softmax(logits[i, sel].float(), dim=-1)
            v, ix = torch.topk(lp, k, dim=-1)
            t = torch.from_numpy(target[a : a + POS_CHUNK]).to(device)
            tgt_lp.append(lp.gather(1, t[:, None])[:, 0].cpu().numpy())
            tops_i.append(ix.cpu().numpy())
            tops_v.append(v.cpu().numpy())
        if len(pos) == 0:
            out.append(
                SeqTopK(
                    s.seq_id, pos, target, np.zeros((0, k), np.int64), np.zeros((0, k)), np.zeros(0)
                )
            )
            continue
        out.append(
            SeqTopK(
                s.seq_id,
                pos.astype(np.int32),
                target,
                np.concatenate(tops_i),
                np.concatenate(tops_v).astype(np.float64),
                np.concatenate(tgt_lp).astype(np.float64),
            )
        )
    return out


def _greedy(model: Any, tok: Any, g: RenderedGreedy, max_new: int, device: Any) -> list[int]:
    import torch

    inp = torch.tensor([g.prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inp,
            attention_mask=torch.ones_like(inp),
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    new = out[0, len(g.prompt_ids) :].tolist()
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    if eos & set(new):
        new = new[: next(i for i, t in enumerate(new) if t in eos)]
    return new


def build_pack_from_objects(
    model: Any,
    tok: Any,
    out_dir: str | Path,
    *,
    repo: str,
    revision: str | None = None,
    dtype: str = "bfloat16",
    probes_version: str = "v1",
    probes_file: Path | None = None,
    k: int = 32,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    logp_dtype: str = "float32",
    license_: str | None = None,
    device: Any = "cpu",
    device_label: str = "cpu",
    progress: Callable[[int, int], None] | None = None,
    max_len: int = 512,
) -> Manifest:
    """Build a pack from an already loaded model and tokenizer."""
    import torch
    import transformers

    probes = load_probes(probes_version, probes_file)
    seqs, greedy = render_probes(probes, tok, max_len=max_len)
    writer = PackWriter(out_dir, k=k, logp_dtype=logp_dtype)
    writer.write_probes(seqs)
    model.eval()

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i].token_ids))
    done = 0
    for a in range(0, len(order), batch_size):
        batch = [seqs[i] for i in order[a : a + batch_size]]
        for r in _topk_batch(model, batch, k, device):
            writer.add_seq(r)
        done += len(batch)
        if progress:
            progress(done, len(seqs) + len(greedy))

    rows = []
    for g in greedy:
        ref = _greedy(model, tok, g, max_new_tokens, device)
        rows.append(
            {
                "seq_id": g.seq_id,
                "group": g.group,
                "lang": g.lang,
                "prompt_ids": g.prompt_ids,
                "ref_ids": ref,
                "ref_text": tok.decode(ref, skip_special_tokens=True),
            }
        )
        done += 1
        if progress:
            progress(done, len(seqs) + len(greedy))
    writer.write_greedy(rows)

    return writer.finalize(
        model=ModelRef(repo=repo, revision=revision, dtype=dtype, license=license_),
        tokenizer=tokenizer_info(tok),
        probes_set=probes_version,
        probes_sha256=_probes_hash(probes_version, probes_file),
        n_sequences=len(seqs),
        chat_template_sha256=chat_template_hash(tok),
        greedy=GreedyInfo(n_prompts=len(greedy), max_new_tokens=max_new_tokens),
        runner=RunnerInfo(
            quantgap=__version__,
            torch=torch.__version__,
            transformers=transformers.__version__,
            device=device_label,
        ),
    )


def _probes_hash(version: str, path: Path | None) -> str:
    if path is None:
        return probes_sha256(version)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(
    repo: str, revision: str | None, device: str = "auto"
) -> tuple[Any, Any, Any, str, str]:
    """Load the reference model. Returns (model, tokenizer, device, dtype_name, device_label)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    use_cuda = device == "cuda" or (device == "auto" and has_cuda)
    if use_cuda and not has_cuda:
        raise RuntimeError("--device cuda was asked for, but no CUDA GPU was found.")
    dtype = torch.bfloat16 if use_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    # `torch_dtype` was renamed `dtype` in newer transformers; pass whichever this version knows.
    import transformers

    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if major_minor >= (4, 56) else "torch_dtype"
    kwargs: dict[str, Any] = {"revision": revision, dtype_key: dtype}
    if use_cuda and torch.cuda.device_count() > 1:
        kwargs["device_map"] = "auto"  # spread across GPUs (Kaggle T4 x2), offload if needed
    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    if "device_map" in kwargs:
        dev = next(model.parameters()).device
        label = f"{torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}"
    elif use_cuda:
        model = model.to("cuda")
        dev = torch.device("cuda")
        label = torch.cuda.get_device_name(0)
    else:
        dev = torch.device("cpu")
        label = "cpu"
    return model, tok, dev, str(dtype).replace("torch.", ""), label
