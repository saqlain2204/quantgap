"""Build a tiny random Llama-style model (+ tokenizer) and, optionally, tiny GGUF files for tests.

    python scripts/make_tiny_model.py [--out tests/fixtures/tiny_model] [--no-gguf]

Output folder:
    hf/           the Hugging Face model and tokenizer (2 layers, hidden 64, vocab 512)
    tiny-f16.gguf the same weights written as F16 GGUF
    tiny-q4_0.gguf the F16 file re-quantized to Q4_0 by llama.cpp's own quantizer

The GGUF writer here is self-contained (uses the `gguf` package) so no llama.cpp checkout or
binary is needed. It is only meant for tiny test models; real GGUFs come from llama.cpp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantgap.tokcheck import CHECK_STRING

VOCAB = 512
HIDDEN, FFN, LAYERS, HEADS, KV_HEADS = 64, 128, 2, 4, 2
MAX_POS = 2048
SPECIALS = ["<s>", "</s>", "<pad>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)
CORPUS = [
    CHECK_STRING,
    "The quick brown fox jumps over the lazy dog. Photosynthesis turns light into sugar.",
    "def add(a, b):\n    return a + b\n\nclass Thing:\n    pass\n",
    '{"name": "Asha", "city": "Pune", "age": 31, "hobbies": ["chess", "cycling"]}',
    "Question: Tom has 3 apples and buys 4 more. How many apples does he have? The answer is 7.",
    "El rapido zorro marron salta sobre el perro perezoso. 你好，世界。 नमस्ते दुनिया.",
]


def build_tokenizer(out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(CORPUS * 20, trainer)
    assert tk.get_vocab_size() == VOCAB, tk.get_vocab_size()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out)
    return fast


def build_model(out: Path, seed: int = 0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=FFN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        max_position_embeddings=MAX_POS,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        initializer_range=0.35,  # bigger than usual so output distributions are not flat
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    model.save_pretrained(out)
    return model


def _permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """HF stores rotary q/k weights in a different order than llama.cpp's GGUF layout."""
    return (
        w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(w.shape)
    )


def write_gguf_f16(model, tok, path: Path) -> None:
    import gguf

    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name("quantgap-tiny")
    w.add_context_length(MAX_POS)
    w.add_embedding_length(HIDDEN)
    w.add_block_count(LAYERS)
    w.add_feed_forward_length(FFN)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(HIDDEN // HEADS)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(VOCAB)
    w.add_file_type(1)  # MOSTLY_F16

    spec = json.loads(tok.backend_tokenizer.to_str())
    vocab = spec["model"]["vocab"]
    tokens = [""] * VOCAB
    for t, i in vocab.items():
        tokens[i] = t
    types = [3 if t in SPECIALS else 1 for t in tokens]
    merges = [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(0)
    w.add_eos_token_id(1)
    w.add_pad_token_id(2)
    w.add_add_bos_token(False)

    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}

    def put(name: str, arr: np.ndarray, f16: bool = True) -> None:
        w.add_tensor(name, arr.astype(np.float16 if f16 else np.float32))

    put("token_embd.weight", sd["model.embed_tokens.weight"])
    put("output_norm.weight", sd["model.norm.weight"], f16=False)
    put("output.weight", sd["lm_head.weight"])
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        put(f"blk.{i}.attn_norm.weight", sd[p + "input_layernorm.weight"], f16=False)
        put(f"blk.{i}.ffn_norm.weight", sd[p + "post_attention_layernorm.weight"], f16=False)
        put(f"blk.{i}.attn_q.weight", _permute(sd[p + "self_attn.q_proj.weight"], HEADS))
        put(f"blk.{i}.attn_k.weight", _permute(sd[p + "self_attn.k_proj.weight"], KV_HEADS))
        put(f"blk.{i}.attn_v.weight", sd[p + "self_attn.v_proj.weight"])
        put(f"blk.{i}.attn_output.weight", sd[p + "self_attn.o_proj.weight"])
        put(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"])
        put(f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"])
        put(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def quantize(src: Path, dst: Path, ftype_name: str = "MOSTLY_Q4_0") -> None:
    """Re-quantize with llama.cpp's own quantizer (shipped inside llama-cpp-python)."""
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_{ftype_name}")
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), params)
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize failed with code {rc}")


def build(out: Path, with_gguf: bool = True) -> Path:
    out = Path(out)
    hf = out / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(hf)
    model = build_model(hf)
    if with_gguf:
        try:
            write_gguf_f16(model, tok, out / "tiny-f16.gguf")
            quantize(out / "tiny-f16.gguf", out / "tiny-q4_0.gguf", "MOSTLY_Q4_0")
        except ImportError as e:
            print(
                f"skipping GGUF files ({e}); install the dev extras: pip install -e .[dev] llama-cpp-python"
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures/tiny_model")
    ap.add_argument("--no-gguf", action="store_true")
    a = ap.parse_args()
    out = build(Path(a.out), with_gguf=not a.no_gguf)
    print("built", out)


if __name__ == "__main__":
    main()
