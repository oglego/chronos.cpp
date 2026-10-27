#!/usr/bin/env python3
"""
Dump golden tensors from a Chronos-Bolt checkpoint for validating the C++ port.

Usage:
    pip install torch transformers numpy chronos-forecasting
    python scripts/dump_golden.py --model amazon/chronos-bolt-tiny --out golden/tiny

Every stage of the forward pass is re-run by hand (using the model's own
submodules) and saved as float32 .npy. The hand-rolled result is then checked
against the model's real forward(), so the golden files are known to be right.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chronos import ChronosBoltPipeline


def save(out: Path, name: str, t) -> None:
    arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
    np.save(out / f"{name}.npy", np.ascontiguousarray(arr))
    print(f"  {name:28s} {str(tuple(arr.shape)):20s} mean={arr.mean():+.4f} std={arr.std():.4f}")


def make_input(length: int, n_leading_nan: int, seed: int) -> torch.Tensor:
    """Deterministic synthetic series. Length is deliberately NOT a multiple of
    the patch size, and leading NaNs exercise the masking path."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    x = 50.0 + 0.1 * t + 10.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 1.0, length)
    x = x.astype(np.float32)
    x[:n_leading_nan] = np.nan
    return torch.from_numpy(x).unsqueeze(0)  # (1, length)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-tiny")
    ap.add_argument("--out", default="golden/tiny")
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--leading-nan", type=int, default=10)
    ap.add_argument("--long-horizon", type=int, default=100, help="prediction_length for the long-horizon test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = ChronosBoltPipeline.from_pretrained(args.model, device_map="cpu", torch_dtype=torch.float32)
    model = pipe.model.eval()
    cc = model.chronos_config
    cfg = model.config

    # ---- 1. Inventory: weights and config (drives the GGUF converter) ----
    print("\n== state_dict ==")
    sd = model.state_dict()
    lines = []
    for k, v in sd.items():
        line = f"{k:70s} {str(tuple(v.shape)):18s} {v.dtype}"
        lines.append(line)
        print(line)
    (out / "state_dict_keys.txt").write_text("\n".join(lines) + "\n")

    meta = {
        "model": args.model,
        "chronos_config": {
            "context_length": cc.context_length,
            "prediction_length": cc.prediction_length,
            "input_patch_size": cc.input_patch_size,
            "input_patch_stride": cc.input_patch_stride,
            "quantiles": list(cc.quantiles),
            "use_reg_token": cc.use_reg_token,
        },
        "t5_config": {
            k: getattr(cfg, k, None)
            for k in [
                "d_model", "d_kv", "d_ff", "num_heads", "num_layers", "num_decoder_layers",
                "dense_act_fn", "is_gated_act", "feed_forward_proj", "layer_norm_epsilon",
                "relative_attention_num_buckets", "relative_attention_max_distance",
                "tie_word_embeddings", "decoder_start_token_id", "vocab_size",
            ]
        },
        "test_input": {"length": args.length, "leading_nan": args.leading_nan, "seed": args.seed},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("\n== config ==")
    print(json.dumps(meta["chronos_config"], indent=2))
    print(json.dumps(meta["t5_config"], indent=2))

    # ---- 2. Stage-by-stage forward pass ----
    print("\n== golden tensors ==")
    context = make_input(args.length, args.leading_nan, args.seed)
    save(out, "00_context", context)  # contains NaNs

    mask = (~torch.isnan(context)).float()
    if context.shape[-1] > cc.context_length:
        context = context[..., -cc.context_length:]
        mask = mask[..., -cc.context_length:]
    save(out, "01_mask", mask)

    # instance norm
    normed, (loc, scale) = model.instance_norm(context)
    save(out, "02_loc", loc)
    save(out, "02_scale", scale)
    save(out, "03_normed", torch.nan_to_num(normed, nan=0.0))  # NaN positions -> 0 for storage

    # patching (pads left with NaN up to a multiple of patch size)
    patched = model.patch(normed)
    patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
    patched = torch.where(patched_mask > 0.0, patched, torch.zeros_like(patched))
    patch_input = torch.cat([patched, patched_mask], dim=-1)  # (B, n_patches, 2 * patch_size)
    attention_mask = patched_mask.sum(dim=-1) > 0
    save(out, "04_patched_values", patched)
    save(out, "05_patched_mask", patched_mask)
    save(out, "06_patch_input", patch_input)
    save(out, "07_attention_mask", attention_mask.float())

    # patch embedding (residual block)
    input_embeds = model.input_patch_embedding(patch_input)
    save(out, "08_input_embeds", input_embeds)

    if cc.use_reg_token:
        reg_ids = torch.full((context.shape[0], 1), cfg.reg_token_id, dtype=torch.long)
        input_embeds = torch.cat([input_embeds, model.shared(reg_ids)], dim=-2)
        attention_mask = torch.cat([attention_mask.float(), torch.ones(context.shape[0], 1)], dim=-1)
        save(out, "09_embeds_with_reg", input_embeds)

    # encoder: dump every layer's output (index 0 = input, last = after final norm)
    enc = model.encoder(
        attention_mask=attention_mask.float(),
        inputs_embeds=input_embeds,
        output_hidden_states=True,
    )
    for i, h in enumerate(enc.hidden_states):
        save(out, f"10_enc_hidden_{i:02d}", h)
    enc_out = enc.last_hidden_state
    save(out, "11_encoder_out", enc_out)

    # relative position bias of encoder layer 0 (the riskiest piece to port)
    try:
        n = input_embeds.shape[1]
        attn0 = model.encoder.block[0].layer[0].SelfAttention
        bias = attn0.compute_bias(n, n)  # (1, n_heads, n, n)
        save(out, "12_enc_rel_pos_bias", bias)
        save(out, "12_enc_rel_bias_weight", attn0.relative_attention_bias.weight)  # (n_buckets, n_heads)
    except Exception as e:  # signature varies across transformers versions
        print(f"  (skipped rel pos bias dump: {e})")

    # decoder: single step from decoder_start_token_id
    dec_out = model.decode(input_embeds, attention_mask, enc_out)  # (B, 1, d_model)
    save(out, "13_decoder_out", dec_out)

    # output head -> normalized quantiles -> unscaled quantiles
    q_norm = model.output_patch_embedding(dec_out).view(
        context.shape[0], model.num_quantiles, cc.prediction_length
    )
    save(out, "14_quantiles_normed", q_norm)
    q_final = model.instance_norm.inverse(q_norm.view(context.shape[0], -1), (loc, scale)).view_as(q_norm)
    save(out, "15_quantiles_final", q_final)

    # ---- 3. Sanity: hand-rolled path must match the real forward() ----
    ref = model(context=context).quantile_preds
    diff = (ref - q_final).abs().max().item()
    print(f"\nmax |forward() - hand-rolled| = {diff:.3e}")
    assert diff < 1e-4, "hand-rolled path diverged from model.forward(); golden tensors are not trustworthy"

    # ---- 4. Long-horizon heuristic (quantile-of-quantiles) ----
    long_ctx = make_input(args.length, args.leading_nan, args.seed)
    long_pred = pipe.predict(long_ctx, prediction_length=args.long_horizon)  # (1, 9, horizon)
    save(out, "20_long_horizon_pred", long_pred)

    # predict_quantiles with a subset of levels and a level that needs interpolation
    q, mean = pipe.predict_quantiles(long_ctx, prediction_length=cc.prediction_length, quantile_levels=[0.1, 0.5, 0.95])
    save(out, "21_pq_quantiles", q)
    save(out, "21_pq_mean", mean)

    print(f"\nDone. Wrote golden tensors to {out}/")


if __name__ == "__main__":
    main()
