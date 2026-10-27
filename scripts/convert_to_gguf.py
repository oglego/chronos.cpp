#!/usr/bin/env python3
"""
Convert a Chronos-Bolt HuggingFace checkpoint to GGUF.

Usage:
    python scripts/convert_to_gguf.py amazon/chronos-bolt-tiny --outfile models/chronos-bolt-tiny-f32.gguf
    python scripts/convert_to_gguf.py models/chronos-bolt-tiny-f32.gguf --outtype f16

Only needs: numpy, gguf, safetensors, huggingface_hub (no torch).

Tensor names follow the llama.cpp T5 scheme, plus two custom blocks:

    token_embd.weight                      shared.weight, (2, d_model): row 0 = decoder start, row 1 = [REG]
    patch_embd.{hidden,output,residual}.{weight,bias}      input_patch_embedding.*
    enc.blk.N.attn_norm.weight             layer.0.layer_norm
    enc.blk.N.attn_{q,k,v,o}.weight        layer.0.SelfAttention.{q,k,v,o}
    enc.blk.0.attn_rel_b.weight            relative_attention_bias, (n_buckets, n_heads); block 0 only, shared by all layers
    enc.blk.N.ffn_norm.weight              layer.1.layer_norm
    enc.blk.N.ffn_up.weight                DenseReluDense.wi   (plain ReLU FFN, not gated)
    enc.blk.N.ffn_down.weight              DenseReluDense.wo
    enc.output_norm.weight                 encoder.final_layer_norm
    dec.blk.N.attn_norm.weight             layer.0.layer_norm
    dec.blk.N.attn_{v,o}.weight            layer.0.SelfAttention.{v,o}
    dec.blk.N.cross_attn_norm.weight       layer.1.layer_norm
    dec.blk.N.cross_attn_{q,k,v,o}.weight  layer.1.EncDecAttention.{q,k,v,o}
    dec.blk.N.ffn_{norm,up,down}.weight    layer.2.*
    dec.output_norm.weight                 decoder.final_layer_norm
    output_patch.{hidden,output,residual}.{weight,bias}    output_patch_embedding.*

Intentionally dropped (never affect the output):
  * encoder.embed_tokens / decoder.embed_tokens: tied duplicates of shared.weight (verified equal)
  * decoder self-attention q, k and relative_attention_bias: the decoder only ever sees ONE token,
    so softmax over a single key is exactly 1 and self-attention reduces to o(v(x)).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import gguf

ARCH = "chronos-bolt"
SKIP = object()  # sentinel: tensor is intentionally not exported

# (regex on HF name, GGUF name template). First match wins.
_RULES = [
    # ---- shared embedding (tied with encoder/decoder embed_tokens) ----
    (r"shared\.weight", "token_embd.weight"),
    (r"(encoder|decoder)\.embed_tokens\.weight", SKIP),  # tied duplicates, checked separately
    # ---- patch embedding / output head (residual blocks) ----
    (r"input_patch_embedding\.(hidden|output|residual)_layer\.(weight|bias)", r"patch_embd.\1.\2"),
    (r"output_patch_embedding\.(hidden|output|residual)_layer\.(weight|bias)", r"output_patch.\1.\2"),
    # ---- encoder ----
    (r"encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.([qkvo])\.weight", r"enc.blk.\1.attn_\2.weight"),
    (r"encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight", r"enc.blk.\1.attn_rel_b.weight"),
    (r"encoder\.block\.(\d+)\.layer\.0\.layer_norm\.weight", r"enc.blk.\1.attn_norm.weight"),
    (r"encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wi\.weight", r"enc.blk.\1.ffn_up.weight"),
    (r"encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.wo\.weight", r"enc.blk.\1.ffn_down.weight"),
    (r"encoder\.block\.(\d+)\.layer\.1\.layer_norm\.weight", r"enc.blk.\1.ffn_norm.weight"),
    (r"encoder\.final_layer_norm\.weight", "enc.output_norm.weight"),
    # ---- decoder: self-attention (only v, o matter for a single token) ----
    (r"decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.([vo])\.weight", r"dec.blk.\1.attn_\2.weight"),
    (r"decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.[qk]\.weight", SKIP),
    (r"decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.relative_attention_bias\.weight", SKIP),
    (r"decoder\.block\.(\d+)\.layer\.0\.layer_norm\.weight", r"dec.blk.\1.attn_norm.weight"),
    # ---- decoder: cross-attention ----
    (r"decoder\.block\.(\d+)\.layer\.1\.EncDecAttention\.([qkvo])\.weight", r"dec.blk.\1.cross_attn_\2.weight"),
    (r"decoder\.block\.(\d+)\.layer\.1\.layer_norm\.weight", r"dec.blk.\1.cross_attn_norm.weight"),
    # ---- decoder: FFN ----
    (r"decoder\.block\.(\d+)\.layer\.2\.DenseReluDense\.wi\.weight", r"dec.blk.\1.ffn_up.weight"),
    (r"decoder\.block\.(\d+)\.layer\.2\.DenseReluDense\.wo\.weight", r"dec.blk.\1.ffn_down.weight"),
    (r"decoder\.block\.(\d+)\.layer\.2\.layer_norm\.weight", r"dec.blk.\1.ffn_norm.weight"),
    (r"decoder\.final_layer_norm\.weight", "dec.output_norm.weight"),
]
_COMPILED = [(re.compile(f"^{pat}$"), tgt) for pat, tgt in _RULES]


def map_name(hf_name: str):
    """Return the GGUF name, SKIP, or raise if the tensor is unknown."""
    for rx, target in _COMPILED:
        m = rx.match(hf_name)
        if m:
            return target if target is SKIP else m.expand(target)
    raise KeyError(f"Unmapped tensor: {hf_name}. Add a rule to _RULES (or to SKIP if it is unused).")


def should_quantize_f16(gguf_name: str) -> bool:
    """With --outtype f16 only the big per-layer matrices are halved. Embeddings, patch/output
    blocks, relative bias, norms and biases stay f32 (tiny, and precision-sensitive)."""
    return (gguf_name.startswith(("enc.blk.", "dec.blk."))
            and gguf_name.endswith(".weight")
            and "norm" not in gguf_name
            and "rel_b" not in gguf_name)


def load_checkpoint(model: str):
    """Returns (config dict, {hf_name: float32 ndarray})."""
    from safetensors.numpy import load_file

    path = Path(model)
    if not path.is_dir():
        if model.startswith((".", "/", "~")) or model.count("/") != 1:
            raise FileNotFoundError(
                f"'{model}' looks like a local path but is not a directory. "
                "Pass an existing checkpoint dir, or a HF repo id like amazon/chronos-bolt-tiny."
            )
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(model, allow_patterns=["config.json", "*.safetensors"]))
    st_files = sorted(path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors in {path} (pytorch_model.bin checkpoints are not supported)")
    state: Dict[str, np.ndarray] = {}
    for f in st_files:
        state.update(load_file(str(f)))
    config = json.loads((path / "config.json").read_text())
    return config, {k: v.astype(np.float32, copy=False) for k, v in state.items()}


def check_config(cfg: dict) -> None:
    """Fail loudly on anything the C++ graph does not implement."""
    assert "chronos_config" in cfg, "Not a Chronos config"
    act = cfg.get("dense_act_fn", "relu")
    assert act == "relu", f"only ReLU is implemented, got {act}"
    assert not cfg.get("is_gated_act", False), "gated FFN is not implemented"
    assert cfg.get("tie_word_embeddings", True), "expected tied embeddings"
    assert cfg["num_layers"] >= 1 and cfg["num_decoder_layers"] >= 1


def convert(config: dict, state: Dict[str, np.ndarray], outfile: Path, outtype: str) -> List[tuple]:
    check_config(config)
    cc = config["chronos_config"]
    n_enc, n_dec = config["num_layers"], config["num_decoder_layers"]
    d_model, d_ff = config["d_model"], config["d_ff"]

    # ---- map tensors, in a stable order ----
    exported: Dict[str, np.ndarray] = {}
    rows = []  # (hf_name, gguf_name or "(dropped)", shape)
    for hf_name in sorted(state):
        target = map_name(hf_name)
        if target is SKIP:
            rows.append((hf_name, "(dropped)", state[hf_name].shape))
            continue
        exported[target] = state[hf_name]
        rows.append((hf_name, target, state[hf_name].shape))

    # tied embeddings must really be duplicates, otherwise dropping them changes the model
    for k in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
        if k in state:
            assert np.array_equal(state[k], state["shared.weight"]), f"{k} differs from shared.weight"

    # ---- completeness check: every tensor the C++ graph will ask for must exist ----
    expected = {"token_embd.weight", "enc.output_norm.weight", "dec.output_norm.weight", "enc.blk.0.attn_rel_b.weight"}
    for part in ("patch_embd", "output_patch"):
        for blk in ("hidden", "output", "residual"):
            expected |= {f"{part}.{blk}.weight", f"{part}.{blk}.bias"}
    for i in range(n_enc):
        expected |= {f"enc.blk.{i}.{n}.weight" for n in
                     ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_o", "ffn_norm", "ffn_up", "ffn_down")}
    for i in range(n_dec):
        expected |= {f"dec.blk.{i}.{n}.weight" for n in
                     ("attn_norm", "attn_v", "attn_o", "cross_attn_norm",
                      "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
                      "ffn_norm", "ffn_up", "ffn_down")}
    missing, extra = expected - exported.keys(), exported.keys() - expected
    assert not missing, f"missing tensors: {sorted(missing)}"
    assert not extra, f"unexpected tensors: {sorted(extra)}"

    # ---- shape sanity ----
    ps = cc["input_patch_size"]
    nq, pl = len(cc["quantiles"]), cc["prediction_length"]
    assert exported["patch_embd.hidden.weight"].shape == (d_ff, 2 * ps)
    assert exported["patch_embd.output.weight"].shape == (d_model, d_ff)
    assert exported["output_patch.output.weight"].shape == (nq * pl, d_ff)
    assert exported["token_embd.weight"].shape[1] == d_model
    assert exported["enc.blk.0.attn_rel_b.weight"].shape == (config["relative_attention_num_buckets"], config["num_heads"])

    # ---- write GGUF ----
    outfile.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(outfile), ARCH)
    w.add_name(config.get("_name_or_path") or "chronos-bolt")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if outtype == "f16" else gguf.LlamaFileType.ALL_F32)

    k = lambda s: f"{ARCH}.{s}"
    # llama.cpp T5-style keys
    w.add_uint32(k("context_length"), cc["context_length"])
    w.add_uint32(k("embedding_length"), d_model)
    w.add_uint32(k("feed_forward_length"), d_ff)
    w.add_uint32(k("block_count"), n_enc)
    w.add_uint32(k("decoder_block_count"), n_dec)
    w.add_uint32(k("attention.head_count"), config["num_heads"])
    w.add_uint32(k("attention.key_length"), config["d_kv"])
    w.add_uint32(k("attention.value_length"), config["d_kv"])
    w.add_float32(k("attention.layer_norm_rms_epsilon"), config["layer_norm_epsilon"])
    w.add_uint32(k("attention.relative_buckets_count"), config["relative_attention_num_buckets"])
    w.add_uint32(k("decoder_start_token_id"), config["decoder_start_token_id"])
    # chronos-bolt specific keys (not in llama.cpp)
    w.add_uint32(k("attention.relative_max_distance"), config["relative_attention_max_distance"])
    w.add_uint32(k("prediction_length"), pl)
    w.add_uint32(k("patch_size"), ps)
    w.add_uint32(k("patch_stride"), cc["input_patch_stride"])
    w.add_array(k("quantiles"), [float(q) for q in cc["quantiles"]])
    w.add_bool(k("use_reg_token"), bool(cc.get("use_reg_token", False)))
    w.add_uint32(k("reg_token_id"), config.get("reg_token_id", 1))
    w.add_string(k("ffn_activation"), "relu")

    for name, arr in exported.items():
        out_arr = arr.astype(np.float16) if (outtype == "f16" and should_quantize_f16(name)) else arr
        w.add_tensor(name, np.ascontiguousarray(out_arr))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    verify(outfile, exported, outtype)
    return rows


def verify(outfile: Path, exported: Dict[str, np.ndarray], outtype: str) -> None:
    """Re-open the written file and compare every tensor to the source."""
    r = gguf.GGUFReader(str(outfile))
    assert bytes(r.fields["general.architecture"].parts[-1]).decode() == ARCH
    got = {t.name: t for t in r.tensors}
    assert got.keys() == exported.keys(), "tensor set mismatch after write"
    for name, src in exported.items():
        data = np.asarray(got[name].data).reshape(src.shape)
        if data.dtype == np.float16:
            np.testing.assert_allclose(data.astype(np.float32), src, rtol=1e-3, atol=1e-3, err_msg=name)
        else:
            np.testing.assert_array_equal(data, src, err_msg=name)
    print(f"verified {len(got)} tensors round-trip OK")


def print_table(rows: List[tuple]) -> None:
    w = max(len(r[0]) for r in rows)
    for hf, g, shape in rows:
        print(f"  {hf:{w}s} -> {g:36s} {tuple(shape)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="HF repo id (e.g. amazon/chronos-bolt-tiny) or local checkpoint dir")
    ap.add_argument("--outfile", type=Path, default=None)
    ap.add_argument("--outtype", choices=["f32", "f16"], default="f32",
                    help="f32 first: lets you get bit-level parity with the golden tensors before shrinking")
    ap.add_argument("--quiet", action="store_true", help="don't print the name-mapping table")
    args = ap.parse_args()

    config, state = load_checkpoint(args.model)
    base = Path(args.model).name if Path(args.model).is_dir() else args.model.split("/")[-1]
    outfile = args.outfile or Path("models") / f"{base}-{args.outtype}.gguf"

    rows = convert(config, state, outfile, args.outtype)
    if not args.quiet:
        print_table(rows)
    n_drop = sum(1 for r in rows if r[1] == "(dropped)")
    print(f"\nwrote {outfile} ({outfile.stat().st_size / 1e6:.1f} MB): "
          f"{len(rows) - n_drop} tensors exported, {n_drop} dropped")


if __name__ == "__main__":
    main()