"""Write a pack: parquet shards plus manifest.json, with a SHA-256 for every file."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quantgap.models import (
    GreedyInfo,
    Manifest,
    ModelRef,
    ProbesInfo,
    RunnerInfo,
    TokenizerInfo,
)
from quantgap.pack.reader import SeqTopK
from quantgap.probes.render import RenderedSeq
from quantgap.util import sha256_file, utc_now

SHARD_ROWS = 40_000
_COMPRESSION = "zstd"


def _fixed_list(values: np.ndarray, width: int, dtype: pa.DataType) -> pa.Array:
    n = len(values)
    offsets = pa.array(np.arange(0, (n + 1) * width, width, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1), type=dtype))


def _ragged_list(rows: list[np.ndarray], dtype: pa.DataType) -> pa.Array:
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))
    flat = np.concatenate(rows) if rows else np.array([])
    return pa.ListArray.from_arrays(offsets, pa.array(flat, type=dtype))


class PackWriter:
    def __init__(self, out_dir: str | Path, k: int, logp_dtype: str = "float32"):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.logp_dtype = logp_dtype
        self._np_dtype = np.float16 if logp_dtype == "float16" else np.float32
        self._pa_dtype = pa.float16() if logp_dtype == "float16" else pa.float32()
        self._buf: list[SeqTopK] = []
        self._buf_rows = 0
        self._shards: list[str] = []
        self.n_positions = 0

    # ---- top-k shards
    def add_seq(self, s: SeqTopK) -> None:
        if s.ids.shape != (len(s.pos), self.k):
            raise ValueError(f"ids must have shape (n_positions, {self.k}), got {s.ids.shape}")
        self._buf.append(s)
        self._buf_rows += len(s.pos)
        self.n_positions += len(s.pos)
        if self._buf_rows >= SHARD_ROWS:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        seq_id = np.concatenate([np.full(len(s.pos), s.seq_id, np.int32) for s in self._buf])
        pos = np.concatenate([s.pos for s in self._buf]).astype(np.int32)
        target = np.concatenate([s.target for s in self._buf]).astype(np.int32)
        ids = np.concatenate([s.ids for s in self._buf]).astype(np.int32)
        logp = np.concatenate([s.logp for s in self._buf]).astype(self._np_dtype)
        lpt = np.concatenate([s.logp_target for s in self._buf]).astype(np.float32)
        table = pa.table(
            {
                "seq_id": pa.array(seq_id, pa.int32()),
                "pos": pa.array(pos, pa.int32()),
                "target": pa.array(target, pa.int32()),
                "ids": _fixed_list(ids, self.k, pa.int32()),
                "logp": _fixed_list(logp, self.k, self._pa_dtype),
                "logp_target": pa.array(lpt, pa.float32()),
            }
        )
        name = f"topk-{len(self._shards):05d}.parquet"
        pq.write_table(table, self.out / name, compression=_COMPRESSION)
        self._shards.append(name)
        self._buf, self._buf_rows = [], 0

    # ---- probes and greedy
    def write_probes(self, seqs: Iterable[RenderedSeq]) -> None:
        seqs = list(seqs)
        table = pa.table(
            {
                "seq_id": pa.array([s.seq_id for s in seqs], pa.int32()),
                "group": pa.array([s.group for s in seqs], pa.string()),
                "token_ids": _ragged_list(
                    [np.array(s.token_ids, np.int32) for s in seqs], pa.int32()
                ),
                "n_tokens": pa.array([len(s.token_ids) for s in seqs], pa.int32()),
                "score_mask": _ragged_list(
                    [np.array(s.score_mask, bool) for s in seqs], pa.bool_()
                ),
            }
        )
        pq.write_table(table, self.out / "probes.parquet", compression=_COMPRESSION)

    def write_greedy(self, rows: list[dict]) -> None:
        """rows: dicts with seq_id, group, lang, prompt_ids, ref_ids, ref_text."""
        table = pa.table(
            {
                "seq_id": pa.array([r["seq_id"] for r in rows], pa.int32()),
                "group": pa.array([r["group"] for r in rows], pa.string()),
                "lang": pa.array([r["lang"] for r in rows], pa.string()),
                "prompt_ids": _ragged_list(
                    [np.array(r["prompt_ids"], np.int32) for r in rows], pa.int32()
                ),
                "ref_ids": _ragged_list(
                    [np.array(r["ref_ids"], np.int32) for r in rows], pa.int32()
                ),
                "ref_text": pa.array([r["ref_text"] for r in rows], pa.string()),
            }
        )
        pq.write_table(table, self.out / "greedy.parquet", compression=_COMPRESSION)

    # ---- manifest
    def finalize(
        self,
        *,
        model: ModelRef,
        tokenizer: TokenizerInfo,
        probes_set: str,
        probes_sha256: str,
        n_sequences: int,
        chat_template_sha256: str | None,
        greedy: GreedyInfo,
        runner: RunnerInfo,
    ) -> Manifest:
        self._flush()
        files = {
            name: sha256_file(self.out / name)
            for name in sorted([*self._shards, "probes.parquet", "greedy.parquet"])
        }
        m = Manifest(
            model=model,
            tokenizer=tokenizer,
            probes=ProbesInfo(
                set=probes_set,
                sha256=probes_sha256,
                n_sequences=n_sequences,
                n_positions=self.n_positions,
            ),
            chat_template_sha256=chat_template_sha256,
            k=self.k,
            logp_dtype=self.logp_dtype,
            greedy=greedy,
            runner=runner,
            files=files,
            created_at=utc_now(),
        )
        text = json.dumps(m.model_dump(by_alias=True, mode="json"), indent=2, ensure_ascii=False)
        (self.out / "manifest.json").write_text(text + "\n", encoding="utf-8")
        return m
