"""Read a pack. Every file is checked against the SHA-256 in the manifest before it is used."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from quantgap.models import Manifest
from quantgap.util import sha256_file


class PackIntegrityError(RuntimeError):
    """A pack file does not match the hash in its manifest (corrupt or edited)."""


@dataclass
class SeqTopK:
    """The reference model's top-k guesses at the scored positions of one sequence."""

    seq_id: int
    pos: np.ndarray  # (n,) position index; predicts token pos + 1
    target: np.ndarray  # (n,) the actual next token
    ids: np.ndarray  # (n, k) reference top-k ids, highest first
    logp: np.ndarray  # (n, k) their natural-log probabilities
    logp_target: np.ndarray  # (n,) reference log probability of the actual next token


@dataclass
class ProbeRow:
    seq_id: int
    group: str
    token_ids: np.ndarray
    score_mask: np.ndarray


@dataclass
class GreedyRow:
    seq_id: int
    group: str
    lang: str
    prompt_ids: list[int]
    ref_ids: list[int]
    ref_text: str


class Pack:
    def __init__(self, root: Path, manifest: Manifest):
        self.root = root
        self.manifest = manifest
        self._topk: dict[int, SeqTopK] | None = None

    @property
    def pack_id(self) -> str:
        return self.manifest.pack_id

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> Pack:
        root = Path(path)
        mf = root / "manifest.json"
        if not mf.is_file():
            raise FileNotFoundError(f"No manifest.json in {root}. Is this a quantgap pack folder?")
        manifest = Manifest.model_validate(json.loads(mf.read_text(encoding="utf-8")))
        pack = cls(root, manifest)
        if verify:
            pack.verify()
        return pack

    def verify(self) -> None:
        for name, want in self.manifest.files.items():
            p = self.root / name
            if not p.is_file():
                raise PackIntegrityError(f"Pack file missing: {name}")
            got = sha256_file(p)
            if got != want:
                raise PackIntegrityError(
                    f"Hash mismatch for {name}: manifest says {want[:12]}..., file is {got[:12]}.... "
                    "The file was changed or the download is broken. Download the pack again."
                )

    def probes(self) -> list[ProbeRow]:
        rows = pq.read_table(self.root / "probes.parquet").to_pylist()
        return [
            ProbeRow(
                r["seq_id"],
                r["group"],
                np.array(r["token_ids"], np.int64),
                np.array(r["score_mask"], bool),
            )
            for r in rows
        ]

    def greedy(self) -> list[GreedyRow]:
        rows = pq.read_table(self.root / "greedy.parquet").to_pylist()
        return [
            GreedyRow(
                r["seq_id"], r["group"], r["lang"], r["prompt_ids"], r["ref_ids"], r["ref_text"]
            )
            for r in rows
        ]

    def topk(self) -> dict[int, SeqTopK]:
        if self._topk is not None:
            return self._topk
        k = self.manifest.k
        parts: dict[int, list[dict[str, np.ndarray]]] = {}
        for name in sorted(n for n in self.manifest.files if n.startswith("topk-")):
            tb = pq.read_table(self.root / name)
            seq = tb["seq_id"].to_numpy()
            pos = tb["pos"].to_numpy()
            target = tb["target"].to_numpy()
            ids = tb["ids"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, k)
            logp = tb["logp"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(-1, k)
            lpt = tb["logp_target"].to_numpy()
            for s in np.unique(seq):
                m = seq == s
                parts.setdefault(int(s), []).append(
                    {"pos": pos[m], "target": target[m], "ids": ids[m], "logp": logp[m], "lpt": lpt[m]}
                )
        out: dict[int, SeqTopK] = {}
        for s, chunks in parts.items():
            cat = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
            order = np.argsort(cat["pos"], kind="stable")
            out[s] = SeqTopK(
                s,
                cat["pos"][order],
                cat["target"][order],
                cat["ids"][order],
                cat["logp"][order].astype(np.float64),
                cat["lpt"][order].astype(np.float64),
            )
        self._topk = out
        return out
