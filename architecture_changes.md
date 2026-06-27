# UViT Architecture Changes (2026-06-27)

Three model-level changes to `models/upos.py`'s `UViT`/`Block`/`Attention`/`CrossAttention`, on top of
the training-pipeline optimizations already covered in `improvement.md`. These modify the network itself,
not the training loop, loss, or sampler — see `improvement.md` §3 for those.

**These are breaking changes to the model's parameter shapes.** Any existing checkpoint
(`workdir/*/ckpts/*.ckpt`) was trained with the old architecture and **cannot be loaded** by the new code —
`state_dict` keys/shapes no longer match (new `adaLN_modulation` params, no more `time_embed`-as-token
usage, new `out_cross_attns`, LayerNorms with `elementwise_affine=False`). Training with these changes
must start from a fresh model (random init), not resume an old run.

---

## 1. QK-Norm in `Attention` and `CrossAttention`

`models/upos.py` — both classes now apply `nn.LayerNorm(head_dim)` to Q and K immediately after splitting
into heads, before the scaled dot product (works in both the xFormers path and the fallback path).

**Why:** This project documented two real activation-overflow incidents (`details.md` §8b — NaN loss
collapse under fp16; §8c — ~48% of training steps silently skipped under fp16 due to forward-activation
overflow, fixed by switching to bf16). QK-norm directly targets the same failure mode — unbounded growth
of attention logits as training progresses — by keeping Q/K magnitudes normalized regardless of how large
the underlying linear projections' outputs get. It's a near-zero-cost addition (two LayerNorms over
`head_dim=64`, negligible compute/params) that should make the network more robust to exactly the kind of
instability already seen in this project, independent of which precision is used.

## 2. AdaLN-Zero timestep conditioning (DiT-style), replacing the time-token

**Before:** the timestep embedding was projected to a single extra token, concatenated onto both the main
sequence (`x`) and the positional-query sequence (`target_pos`) before entering `in_blocks`. Every layer
had to learn, via plain self-attention, to "notice" that one token among 144 others.

**After:** `Block` now computes 6 modulation vectors (`shift1, scale1, gate1, shift2, scale2, gate2`) from
the timestep embedding via a small `SiLU → Linear` MLP (`adaLN_modulation`), and uses them to scale/shift
each sub-layer's `LayerNorm` output and gate its residual contribution:

```
shift1, scale1, gate1, shift2, scale2, gate2 = adaLN_modulation(c).chunk(6, dim=-1)
x = x + gate1 * attn(modulate(norm1(x), shift1, scale1))
x = x + gate2 * mlp(modulate(norm2(x), shift2, scale2))
```

`norm1`/`norm2` (and the final `norm` before `decoder_pred`) are now `elementwise_affine=False` — AdaLN
takes over the scale/shift role LayerNorm's own weight/bias used to play. The final `adaLN_modulation`
Linear in every block (and in the new `final_modulation` before `decoder_pred`) is explicitly zero-initialized
*after* the model's generic `trunc_normal_` init pass (the zero-init has to happen after `self.apply(self._init_weights)`,
otherwise that pass overwrites it) — so at initialization every block is a pure residual pass-through
(`gate=0`), and the network gradually learns how much each block should respond to the diffusion timestep.

Net effect on data flow: `x` and `target_pos` are no longer extended by one token anywhere in the network —
they stay at their natural `144`-token length throughout, and the old `x[:, -L:, :]` slice used to strip
the time token off the final output before `unpatchify` is no longer needed (removed).

**Why:** Token-based timestep conditioning is the pre-DiT (2022) convention this paper's UViT base used.
AdaLN-Zero, introduced in Peebles & Xie's *DiT* (arXiv:2212.09748), is now the standard mechanism in
diffusion transformers and is reported to converge faster and more stably than token-concatenation —
directly relevant given this project's own convergence-plateau diagnosis (`improvement.md` §1).
Zero-initialization specifically targets early-training stability (gradients can't blow up out of the gate
when every new sub-layer initially contributes nothing).

## 3. Multi-point cross-attention injection in the decoder

**Before:** the positional query (`target_pos`) only met the encoder's output once, at the single
bottleneck `CrossAttention` call between `in_blocks` and `mid_block`. The 10 `out_blocks` afterward never
see it again directly (only indirectly, via skip connections from `in_blocks`, which themselves only saw it
transitively through earlier layers).

**After:** a new `nn.ModuleList` of 5 additional `CrossAttention` modules (`out_cross_attns`) is interleaved
into the decoder — applied **residually** (added to `x`, not replacing it) after every other `out_block`
(indices 1, 3, 5, 7, 9 of the 10 decoder blocks):

```python
for i, blk in enumerate(self.out_blocks):
    x = blk(x, c, skips.pop())
    if i % 2 == 1:
        x = x + self.out_cross_attns[i // 2](x, target_pos)
```

The original bottleneck `cross_attn` call is unchanged (still a full replacement of `x`, preserving its
original semantics as the primary encoder→decoder positional handoff).

**Why:** gives the decoder repeated, direct access to the positional-query signal instead of relying on it
propagating through 10+ self-attention layers via skip connections alone. Should matter most for the
largest, most geometrically demanding outpainting ratios (e.g. the paper's 11.7× setting), where the
relationship between anchor and target view is least trivial for the network to infer indirectly.

---

## Parameter count and memory impact — important caveat

Measured directly (`embed_dim=1024, depth=20, num_heads=16`, matching `configs/flickr192_*`):

| | Param count |
|---|---|
| Original architecture (per `details.md` §8a) | ~289.8M |
| **With these three changes** | **~445.1M (+54%)** |

The increase is almost entirely from `adaLN_modulation` (`Linear(1024, 6144)` × 21 blocks ≈ 132M params) and
the 5 new `out_cross_attns` (≈ 21M params) — QK-norm itself adds a negligible amount.

**This directly conflicts with the hard 8GB-VRAM constraint this whole project has been built around.**
`improvement.md` §5.5 found the *previous*, smaller model already left only ~340MB of headroom after
dropping EMA and switching to 8-bit AdamW. A 54% larger raw model (and proportionally larger AdamW moment
buffers) will very likely not fit in the budget that `flickr192_optimized.py` was tuned for — **this needs
to be re-verified empirically (a short dry run, watching `nvidia-smi`) before committing to a long training
run.** If it doesn't fit, the cheapest lever is shrinking the AdaLN MLP's hidden width (e.g. project the
timestep embedding to a smaller dim before the `6*dim` modulation Linear) rather than reverting the
mechanism entirely.

## Verification performed

- Forward + backward pass tested at both a tiny config (`embed_dim=64, depth=4`) and the real
  `flickr192_*` config size (`embed_dim=1024, depth=20, num_heads=16`) — correct output shape
  `(B, 4, 24, 24)`, all-finite output, all parameters receive gradients.
- Re-tested with `use_checkpoint=True` (the setting actually used in every `configs/*.py` in this repo) —
  gradient checkpointing still works correctly through the new `Block.forward(x, c, skip)` signature.
- **Not yet tested:** an actual multi-step training run, VRAM fit on the real 8GB target GPU, or any
  FID/IS/PSNR comparison against the pre-change baseline. Recommended next step before relying on this
  architecture for real results.
