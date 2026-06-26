# PQDiff — Architecture & Training Guide

**Paper:** [Continuous-Multiple Image Outpainting in One-Step via Positional Query and A Diffusion-based Approach](https://arxiv.org/abs/2401.15652) (ICLR 2024)

---

## 1. System Overview

PQDiff performs **conditional image outpainting** using a two-stage latent diffusion framework:

```
Input image
    │
    ├─▶ [Frozen VAE Encoder]  →  latent z  (4 × 24 × 24)
    │                                │
    │                           [UViT Denoiser] ← diffusion timestep t
    │                                │           ← anchor view (context)
    │                                │           ← target positional query
    │                                ▼
    └─▶ [Frozen VAE Decoder] ← denoised latent  →  outpainted image (192 × 192)
```

---

## 2. Model Architecture

### 2.1 Frozen Autoencoder (VAE)

File: [libs/autoencoder.py](libs/autoencoder.py)

Ported from Stable Diffusion's KL-regularised autoencoder. **Weights are frozen** during training.

| Property | Value |
|---|---|
| Weights | `assets/stable_diffusion/autoencoder_kl.pth` |
| Input resolution | 192 × 192 × 3 |
| Latent shape | 4 × 24 × 24 (8× spatial compression) |
| Scale factor | 0.18215 |
| Architecture | Conv encoder + Conv decoder with ResNet blocks and attention |
| Channel multipliers | [1, 2, 4, 4] |
| z channels | 4 |

The encoder produces a 4-channel latent; the decoder reconstructs pixel-space images from denoised latents.

### 2.2 UViT Denoiser — `UViT` in `models/upos.py`

The core trainable network. It is a **U-shaped Vision Transformer** adapted for positional-query-guided outpainting.

File: [models/upos.py](models/upos.py)

#### Input preparation

The anchor (context) view and the noisy target latent are **channel-concatenated** before patch embedding:

```
anchor_latent  (B, 4, 24, 24)
noisy_target   (B, 4, 24, 24)
───────────────────────────────
concat input   (B, 8, 24, 24)
```

#### Patch embedding

| Property | Value |
|---|---|
| Latent image size | 24 × 24 |
| Patch size | 2 × 2 |
| Number of patches | 12 × 12 = **144** |
| Embedding dim | **1024** |
| Input channels (to PatchEmbed) | 8 (4 anchor + 4 noisy target) |

#### Positional & timestep tokens

- **Absolute positional embeddings**: 2D sin-cos, fixed (non-learnable), shape `(1, 144, 1024)`
- **Target positional query**: 2D local sin-cos encoding of the *relative crop geometry* between anchor and target views, shape `(B, 144, 1024)`. A learnable `masked_embed` is added.
- **Time token**: Sinusoidal timestep embedding (dim=1024, identity projection), prepended as a single token → sequence length becomes `1 + 144 = 145`

#### Transformer body (depth=20)

```
in_blocks  (10 × Block)        ← encoder arm
      ↓  store skip connections
CrossAttention(x, target_pos)  ← positional query injection
      ↓
mid_block  (1 × Block)
      ↓
out_blocks (10 × Block)        ← decoder arm, U-Net skip connections
      ↓
LayerNorm
      ↓
Linear(1024 → 16)              ← patch_dim = patch_size² × in_chans = 4×4 = 16
      ↓
unpatchify → (B, 4, 24, 24)
      ↓
Conv2d(4, 4, 3, padding=1)     ← final refinement
```

Each `Block` consists of:
- `LayerNorm → Multi-head Self-Attention → residual`
- `LayerNorm → MLP (hidden dim = 4096, GELU) → residual`
- Decoder blocks additionally have a `skip_linear(2048 → 1024)` for U-Net skip fusions

The `CrossAttention` layer uses the target positional query as *queries* and the encoder output as *keys/values*, injecting spatial outpainting intent before the bottleneck.

#### Key hyperparameters (intel192_local.py)

| Parameter | Value |
|---|---|
| embed_dim | 1024 |
| depth | 20 (10 enc + 1 mid + 10 dec) |
| num_heads | 16 (head_dim = 64) |
| mlp_ratio | 4 |
| qkv_bias | False |
| mlp_time_embed | False (Identity) |
| num_classes | 1001 (1000 real + 1 null for CFG) |
| use_checkpoint | **True** (gradient checkpointing — essential for 8 GB GPUs) |

Estimated trainable parameters: **~400–500 M**

---

## 3. Diffusion Process

### 3.1 SDE Formulation

File: [sde.py](sde.py)

PQDiff uses a **Variance Preserving SDE (VPSDE)**:

```
dx = -½ β(t) x dt  +  √β(t) dw

β(t) = β₀ + t(β₁ − β₀),   β₀=0.1,  β₁=20,   t ∈ [0, 1]
```

The marginal distribution is:

```
q(xₜ | x₀) = N(√ᾱ(t) x₀,  (1−ᾱ(t)) I)
```

Training objective: **noise prediction** (`pred = 'noise_pred'`), minimising:

```
L_simple = E_{t, ε, x₀} [ ‖ε − εθ(xₜ, conditions, t)‖² ]
```

### 3.2 Sampling

- **Algorithm**: DPM-Solver (fast ODE solver), 50 steps
- **Guidance**: Classifier-Free Guidance (CFG) with scale=0.4
- EMA model used at inference (rate=0.9999)

---

## 4. Dataset — Flickr Scenery (current)

> **The Intel Image Classification dataset was decommissioned on 2026-06-19.** It produced a model that
> trained to 500,000 steps but the user found its outputs unsatisfactory ("not behaving properly"). All
> Intel-related artifacts were deleted to reclaim disk space — see [§4a](#4a-intel-dataset-decommissioned)
> for what was removed. The project now trains on **Flickr Scenery**, the exact dataset used in the
> original PQDiff paper.

Loader: `Flickr` class in [dataset/dataset.py:204-240](dataset/dataset.py#L204-L240) — already implemented, no code changes needed.

Config path (must end with `/`): `./dataset/scenery/train/`

**Filename requirement:** the loader filters files by a numeric suffix in the filename:

```python
train_files = [path+f for f in os.listdir(path) if int(f.split('_')[-1].split('.')[0].replace(',', '')) <= 5040]
```

Every image filename must end in `_<number>.jpg` (or similar), e.g. `scenery_1024.jpg`, and only files with
that trailing number ≤ 5040 are used. This matches the original QueryOTR Flickr Scenery dataset's naming
convention out of the box — if you obtain images from elsewhere, you'll need to rename them to fit this
pattern (or sequentially rename on import, see [§7.4](#74-importing-the-dataset-from-your-local-computer-to-this-server)).

Each training sample is constructed **online** from a single image (same recipe as before, dataset-agnostic):

| Step | Detail |
|---|---|
| Anchor crop | Random 15–40% of image area (smaller "visible" window) |
| Target crop | Random 80–100% of image area (larger "full" view) |
| Positional query | Relative sin-cos embedding encoding where the target is in relation to the anchor |
| Resolution | Both crops resized to 192 × 192 |
| Normalisation | `[0, 255] → [-1, 1]` |

Sample tuple returned: `(target_img, anchor_img, target_pos)`

### 4a. Intel dataset (decommissioned)

Removed on 2026-06-19, freeing ~2.0 TB:

| Path | Size | Action |
|---|---|---|
| `workdir/intel192_local/` | 2.0 TB | deleted |
| `workdir/intel192_large/` | 84 KB | deleted |
| `dataset/intel/` | 400 MB | deleted |
| `configs/intel192_local.py` | — | deleted |
| `configs/intel192_large.py` | — | deleted |

The `Intel` dataset-loader class itself ([dataset/dataset.py:242-280](dataset/dataset.py#L242-L280)) was left in place — it's generic, reusable code with no dependency on the deleted data, in case Intel-style class-foldered datasets are useful again later.

The technical incidents in [§8a](#8a-why-multi_gpu-oom-on-these-cards) through [§8d](#8d-redoing-the-400000500000-range-under-bf16) were diagnosed on the Intel run, but the **fixes are dataset-agnostic and remain active**: single-GPU launch via plain `python`, the finite-loss guard + gradient clipping in `train_ldm.py`, and the fp16→bf16 switch all apply to Flickr training too.

---

## 5. Training Configuration — `configs/flickr192_local.py`

New config, mirroring the proven `intel192_local.py` settings (same model size, same single-GPU memory budget) but pointed at the Flickr dataset. One deliberate change: **`save_interval` raised from 2,000 to 10,000 steps**.

| Parameter | Value |
|---|---|
| Seed | 1234 |
| Total steps | 500,000 |
| Batch size | 4 |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.03, β=(0.99, 0.99) |
| LR scheduler | Linear warmup for 1,000 steps, then constant |
| Log interval | every 10 steps |
| Eval interval | every 1,000 steps (saves sample grid) |
| **Save interval** | **every 10,000 steps** (was 2,000 for Intel) |
| Mixed precision | bf16 (set via `ACCELERATE_MIXED_PRECISION` env var, not in this file — see §8c) |

**Why the save_interval change:** each checkpoint is ~4.4 GB (raw model + EMA + optimizer state). At the old
2,000-step interval, a full 500,000-step run produces 250 checkpoints (~1.1 TB) — exactly what filled the
disk on the Intel run. At 10,000 steps, the same run produces only 50 checkpoints (~220 GB), a much more
sustainable footprint. Adjust `config.train.save_interval` in `configs/flickr192_local.py` if you want a
different trade-off between recovery granularity and disk usage.

---

## 6. GPU Setup

Hardware available on this machine:

| GPU | Model | VRAM | Current Use |
|---|---|---|---|
| GPU 0 | NVIDIA GeForce RTX 2070 | 8192 MiB | ~512 MiB (display/Xorg) |
| GPU 1 | NVIDIA GeForce RTX 2070 | 8192 MiB | ~11 MiB (idle) |

> **Use GPU 1, single process only.** [§8a](#8a-why-multi_gpu-oom-on-these-cards) confirmed that splitting
> this model across both GPUs via `--multi_gpu` causes a CUDA OOM — DDP's per-GPU overhead (model + EMA +
> optimizer state + gradient-bucket buffers) exceeds 8 GB before the VAE even loads. GPU 1 is preferred over
> GPU 0 simply because it has no desktop/Xorg processes competing for VRAM.

Gradient checkpointing (`use_checkpoint=True`) trades compute for memory, making ~400–500 M parameter training feasible on 8 GB GPUs.

---

## 7. Prerequisites

### 7.1 Virtual environment

```bash
cd /mnt/zone_a/PQDiff/PQDiff
source .venv/bin/activate
```

### 7.2 Required assets (already present)

```
assets/stable_diffusion/autoencoder_kl.pth   ✅ present
```

### 7.3 Dataset

Place Flickr Scenery images flat (no subfolders) at:

```
./dataset/scenery/train/
├── scenery_1.jpg
├── scenery_2.jpg
├── ...
└── scenery_5040.jpg
```

Source: the QueryOTR repo linked from this project's [README.md](README.md#dataset-preparing) ("We use Flickr, Buildings, and WikiArt datasets, which can be obtained at [link](https://github.com/Kaiseem/QueryOTR)"). Remember the filename convention from [§4](#4-dataset--flickr-scenery-current): files must end in `_<number>.jpg`, and only numbers ≤ 5040 are used by the loader.

### 7.4 Importing the Dataset from Your Local Computer to This Server

This server is reachable over the network as:

```
hostname: cvig20
IP:       10.0.116.156
user:     project
target:   /mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train/
```

(Confirm with whoever manages this machine that `10.0.116.156` is reachable from your laptop — it may only be visible on an internal LAN/VPN.)

#### Option A — `scp` (simple, one-shot)

From your **local computer's** terminal (not this server):

```bash
# create the destination directory first (run on the server, e.g. via this same session)
mkdir -p /mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train/

# then from your LOCAL machine:
scp -r /path/to/your/scenery_images/* project@10.0.116.156:/mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train/
```

#### Option B — `rsync` (recommended for large datasets)

`rsync` resumes interrupted transfers and shows progress — much better than `scp` for thousands of images:

```bash
# from your LOCAL machine:
rsync -avz --progress /path/to/your/scenery_images/ \
    project@10.0.116.156:/mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train/
```

If the transfer drops partway through, just re-run the same command — `rsync` skips files already copied.

#### Option C — compress first, then transfer (best for many small files)

Thousands of individual small JPEGs transfer much faster as one archive than as separate SCP/rsync file
operations (each file has connection overhead). On your **local machine**:

```bash
tar -czf scenery.tar.gz -C /path/to/your/scenery_images .
scp scenery.tar.gz project@10.0.116.156:/mnt/zone_a/PQDiff/PQDiff/dataset/
```

Then on the **server** (this session):

```bash
cd /mnt/zone_a/PQDiff/PQDiff/dataset
mkdir -p scenery/train
tar -xzf scenery.tar.gz -C scenery/train
rm scenery.tar.gz
```

#### After transfer: verify the filename convention

```bash
cd /mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train
ls | head -5
# every filename must match: <anything>_<number>.<ext>, number <= 5040 to be used
```

If your filenames don't already follow `<name>_<number>.jpg`, rename them sequentially before training:

```bash
cd /mnt/zone_a/PQDiff/PQDiff/dataset/scenery/train
i=1
for f in *.jpg; do
    mv "$f" "scenery_${i}.jpg"
    i=$((i+1))
done
```

This sequential rename only works correctly if you have ≤ 5040 images you want to keep (all will pass the
loader's filter) — if you have more, only the first 5040 (by this renaming order) will be used for training.

---

## 8. Running Training

> **Important:** Always use `train_ldm.py`, not `train.py`.
> `train_ldm.py` encodes images through the VAE before passing 4×24×24 latents to the UViT.
> `train.py` skips encoding and sends raw 192×192 pixels to the model, causing an immediate shape assertion error.

> **Use single GPU, not `--multi_gpu`.** Tested and confirmed on the Intel run: `--multi_gpu --num_processes 2`
> OOMs on these cards. See [§8a](#8a-why-multi_gpu-oom-on-these-cards) for the root cause — this finding is
> dataset-agnostic and applies equally to Flickr training.

> **Use plain `python`, not `accelerate launch`, for single GPU.** This machine has a saved accelerate
> config at `~/.cache/huggingface/accelerate/default_config.yaml` with `distributed_type: MULTI_GPU` and
> `num_processes: 2` baked in. `accelerate launch` picks up that file even when you pass `--num_processes 1`
> on the command line, and still invokes the `multi_gpu_launcher` path — which then conflicts with
> `CUDA_VISIBLE_DEVICES=1` and fails. Running the script directly with `python` skips that config file
> entirely; `accelerate.Accelerator()` just auto-detects the single visible GPU.

> **Use `bf16`, not `fp16`.** [§8c](#8c-incident-48-of-steps-skipped-on-fp16--switched-to-bf16) found that
> fp16 silently skipped ~48% of training steps once the model was well-converged, due to forward-activation
> overflow (fp16's max representable value is only ~65,504). bf16 has fp32's exponent range and doesn't hit
> this ceiling. Starting Flickr training in bf16 from step 0 avoids ever hitting this issue.

### Single GPU, bf16 (recommended)

```bash
cd /mnt/zone_a/PQDiff/PQDiff
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
    --config=configs/flickr192_local.py
```

- `CUDA_VISIBLE_DEVICES=1` targets the GPU with no desktop/Xorg processes competing for VRAM (GPU 1 has ~11 MiB used vs. ~518 MiB on GPU 0).
- `ACCELERATE_MIXED_PRECISION=bf16` is required because plain `python` (no `accelerate launch`) doesn't read the saved config file's `mixed_precision` setting — without this env var, `Accelerator()` silently falls back to full fp32, using more VRAM.

### Custom workdir

```bash
CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
    --config=configs/flickr192_local.py \
    --workdir=workdir/flickr192_local/run1
```

### Convenience wrapper script

`train_resume.sh` (already in the repo root) bundles the W&B fixed-run-ID env vars (§9a) with this working command:

```bash
chmod +x train_resume.sh
./train_resume.sh
```

---

## 8a. Why `--multi_gpu` OOMs on these cards

Reproduced on 2026-06-17 — full traceback in the crash log:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 7.60 GiB of which 47.50 MiB is free.
... this process has 6.95 GiB memory in use.
  File "train_ldm.py", line 66, in train
    autoencoder.to(device)
```

The crash happens while loading the **frozen VAE** onto the GPU — i.e. the UViT model alone (plus its EMA copy, optimizer state, and DDP overhead) already exhausts available VRAM before the VAE even gets a chance to load.

Per-GPU memory floor with `nnet` = 289,834,148 params (fp32):

| Component | Approx. size |
|---|---|
| Raw UViT | 1.16 GB |
| EMA UViT | 1.16 GB |
| AdamW optimizer state (2 moment buffers, raw model only) | 2.32 GB |
| DDP gradient-bucket replica buffers (NCCL) — **multi-GPU only** | ~1.2 GB |
| **Subtotal before VAE / activations** | **~6.95 GB** |
| Frozen VAE (~85M params) + activations | needs ~0.5–1 GB more → **OOM** |

Switching to `--multi_gpu --num_processes 2` only halves the *activation* memory (mini-batch 4→2 per GPU) — and activations are already minimized by `use_checkpoint=True`. It does **not** shrink the much larger fixed cost (model + EMA + optimizer state), and it *adds* DDP communication buffers that single-GPU training never needs. Net effect: multi-GPU mode increases the per-GPU memory floor, making OOM worse, not better, for this model size on 8 GB cards.

**Conclusion:** single-GPU is not just simpler here — it is the only configuration that fits in 8 GB for this model size. This matches the fact that the existing 400,000-step checkpoint was trained single-GPU.

---

## 8b. Incident: NaN Loss Collapse After Resuming (step 400,190) — Fixed

### What happened

After resuming from `400000.ckpt` on 2026-06-17, the loss went `nan` at **step 400190** (19 seconds into the run) and was permanently `nan` for essentially the rest of training, through step 500,000. This silently corrupted every checkpoint saved after that point and ran undetected for ~7 hours.

Verified by inspecting the actual checkpoint weights:

```python
# 400000.ckpt — clean
nnet bad tensors: 0 / 266

# 402000.ckpt — corrupted (saved 2000 steps after the NaN appeared)
nnet bad tensors:     265 / 266  (NaN/Inf)
nnet_ema bad tensors: 265 / 266  (NaN/Inf)
```

This is why every `predict_target-*.png` from step ≥402000 onward is solid black: `DatasetFactory.unpreprocess()` ([dataset/dataset.py:78-81](dataset/dataset.py#L78-L81)) does `0.5*(v+1)` then `.clamp_(0, 1)` — NaN/Inf pixel values collapse to 0 (black) after clamping.

### Root cause

`train_ldm.py` trains in fp16 mixed precision but, unlike `train.py`, had **no gradient clipping and no finite-loss check** ([train_ldm.py:89-107](train_ldm.py#L89-L107), pre-fix). A single fp16 overflow in the forward pass (activations exceeding fp16's ~65504 max — plausible right after a fresh process resets PyTorch's `GradScaler` to its default high initial scale) produced a `nan` loss. Because there was no guard, the optimizer still applied that batch's update, and `train_state.ema_update()` ([utils.py:92-94](utils.py#L92-L94)) unconditionally blended the now-NaN raw model into the EMA model every step. Once weights go NaN, every later forward pass is NaN too — there's no self-recovery, and the training loop has no check to stop or warn.

### Fix applied

1. **Restored the checkpoint chain.** Moved all checkpoints after 400000 (`402000.ckpt` … `500000.ckpt`, 50 total) to `workdir/intel192_local/x0pred/ckpts_corrupted_archive/`. `train_state.resume()` ([utils.py:110-124](utils.py#L110-L124)) picks the highest-numbered checkpoint in `ckpts/`, so it now correctly resumes from the last verified-clean state: **`400000.ckpt`**.

2. **Patched `train_ldm.py`'s `train_step`** ([train_ldm.py:89-110](train_ldm.py#L89-L110)) with two safeguards:

   ```python
   _metrics['loss'] = accelerator.gather(loss.detach()).mean()
   if not torch.isfinite(_metrics['loss']):
       logging.warning(f'step {train_state.step}: non-finite loss, skipping update')
       train_state.step += 1
       return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)
   accelerator.backward(loss.mean())
   accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.get('grad_clip', 1.0))
   optimizer.step()
   ...
   ```

   - **Finite-loss guard**: if a batch produces `nan`/`inf` loss, the step is skipped entirely — no optimizer update, no EMA blend — so a single bad batch can no longer permanently corrupt the model. It just logs a warning and moves to the next batch.
   - **Gradient clipping** (`max_norm=1.0`, matching `train.py`'s existing pattern): reduces the chance of a large gradient spike pushing fp16 activations out of range in the first place.

### Resuming safely now

Same command as before — no changes needed to how you launch it:

```bash
./train_resume.sh
```

It will now resume from `400000.ckpt` and skip (rather than crash on) any future non-finite-loss batch.

### Resume from checkpoint (automatic)

Training resumes automatically from the latest `.ckpt` in `workdir/<name>/ckpts/`. No extra flag needed.

---

## 8c. Incident: ~48% of Steps Skipped on fp16 — Switched to bf16

### What happened

The 8b guard worked — `500000.ckpt` finished completely clean (0 corrupted tensors) — but the actual run that produced it logged **47,840 `non-finite loss, skipping update` warnings out of ~100,000 total batches (≈48%)** between resuming at 400,000 and finishing at 500,000. The guard correctly prevented corruption, but it means roughly half of that entire 100k-step training stretch did nothing — fp16's overflow rate at this late, well-converged stage of training was far too high to just patch around with skip-and-continue.

### Root cause

fp16 has a maximum representable value of ~65,504. By step ~400k the model is well-converged and produces larger/more confident activations (e.g. sharper attention logits, larger pre-softmax values) — these increasingly exceed fp16 range, triggering overflow in the forward pass at a high, persistent rate. This isn't something gradient clipping or a fresh `GradScaler` can fix, since the overflow happens in forward activations, not just gradients.

### Fix: switch to bf16

bf16 has the **same exponent range as fp32** (no practical overflow risk for this kind of value), at the cost of fewer mantissa bits (less precision per number, which has negligible effect on diffusion training).

**Hardware check:** RTX 2070 is Turing (compute capability 7.5). bf16 lacks dedicated tensor-core acceleration on Turing (that requires Ampere/sm_80+), but `torch.cuda.is_bf16_supported()` returns `True` and it runs correctly via CUDA cores — confirmed empirically:

```bash
CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
    --config=configs/intel192_local.py --config.train.n_steps=500300 \
    --workdir=workdir/_bf16_resume_test   # isolated test copy of 500000.ckpt
```

Result: **80/80 steps valid, 0 non-finite losses** — resumed from the exact same real, late-stage checkpoint that produced a 48% skip rate under fp16.

`train_resume.sh` now uses `ACCELERATE_MIXED_PRECISION=bf16` instead of `fp16`:

```bash
CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
    --config=configs/intel192_local.py
```

### Trade-off to be aware of

Without tensor-core acceleration for bf16 matmuls on Turing, throughput may be modestly slower than fp16 was *when fp16 wasn't overflowing*. In practice this doesn't matter here — fp16 was already wasting ~48% of steps on no-op skips, so bf16's full step utilization easily outweighs any per-step slowdown.

---

## 8d. Redoing the 400,000→500,000 Range Under bf16

The fp16 run that produced the 8c findings did finish "successfully" (clean final weights), but two things in that range are unsalvageable on their own and were redone from scratch:

1. **~48% of training steps in that range were no-ops** (skipped due to non-finite loss) — the model only received genuine gradient updates for roughly half of those 100,000 steps.
2. **Some periodic eval sample images are corrupted**, even though the saved weights are clean — e.g. `predict_target-480001.png` is 523 bytes (a blank/solid-color image) versus a normal ~200–300 KB photo-content PNG. This happens because the *sampling* forward pass at eval time runs under the same fp16 autocast and can overflow independently of whether the stored weights are NaN-free.

### Procedure

1. **Archive the fp16-era checkpoints**, restoring `400000.ckpt` as the latest:
   ```bash
   cd workdir/intel192_local/x0pred
   mkdir -p ckpts_fp16_partial_archive
   for d in $(ls ckpts | sed 's/\.ckpt//' | sort -n); do
     [ "$d" -gt 400000 ] && mv "ckpts/${d}.ckpt" ckpts_fp16_partial_archive/
   done
   ```
   (Not deleted — kept for reference/comparison if needed.)

2. **`train_resume.sh` already points at bf16** (§8c) and now uses a **new W&B run ID** (`intel192_local_x0pred_v3_bf16`) so the redo doesn't log duplicate/overlapping data points over the same step range in the existing `v2` run's history:
   ```bash
   export WANDB_RUN_ID=intel192_local_x0pred_v3_bf16
   ...
   CUDA_VISIBLE_DEVICES=1 ACCELERATE_MIXED_PRECISION=bf16 python train_ldm.py \
       --config=configs/intel192_local.py
   ```

3. **Just run it:**
   ```bash
   ./train_resume.sh
   ```
   It resumes from `400000.ckpt` and re-executes steps 400,001→500,000. Sample images are saved under the same filenames as before (`predict_target-400001.png`, `predict_target-401001.png`, …, `predict_target-499001.png`), so each one is **overwritten in place** with a fresh, bf16-generated image as training passes through that step again — no manual cleanup of `samples/` needed.

4. **Time estimate:** at the observed ~1.7–2.3 it/s, 100,000 steps is roughly 12–16 hours. Run it in the background (`nohup ./train_resume.sh > /tmp/train_v3.log 2>&1 &` or equivalent) and check `output.log` / sample file sizes periodically — a healthy run should show no `non-finite loss` warnings at all (confirmed in the §8c bf16 test) and sample PNGs consistently in the 150–300 KB range, not a few hundred bytes.

---

## 9. Monitoring

> Note: with `train_ldm.py` and no `--config.xxx=...` overrides on the command line, `get_hparams()` ([train_ldm.py:198-212](train_ldm.py#L198-L212)) defaults the run folder name to **`x0pred`**, not `default`. The real run lives at `workdir/flickr192_local/x0pred/`.

Logs are written to `workdir/flickr192_local/x0pred/output.log`.

WandB is used in **offline mode** by default:

```bash
# View offline runs
wandb sync workdir/flickr192_local/x0pred/wandb/offline-run-*/
```

Sample grid images are saved every 1,000 steps to `workdir/flickr192_local/x0pred/samples/`:

- `predict_target-<step>.png` — model output
- `decode_target-<step>.png` — target re-decoded through VAE (sanity check)
- `prime_target-<step>.png` — ground truth target
- `decode_anchor-<step>.png` — anchor re-decoded through VAE
- `prime_anchor-<step>.png` — input anchor view

---

## 9a. Resetting a Messed-Up W&B History

> This section documents a historical incident on the (now-deleted) Intel run. The **principle still
> applies and is already in effect** for Flickr training: `train_resume.sh` pins
> `WANDB_RUN_ID=flickr192_local_x0pred_v1_bf16` from the very first launch, so the Flickr W&B history
> starts clean and continuous — it will not fragment the way the Intel run's history did below.

### What went wrong

`train_ldm.py` calls `wandb.init(..., mode='offline')` on **every process launch** ([train_ldm.py:44](train_ldm.py#L44)) with no fixed run `id`. Each time training was stopped and restarted (crash, manual stop, resuming from a checkpoint), W&B silently started a **brand-new, disconnected run** instead of continuing the previous one. The result in this project's history:

```bash
$ ls workdir/intel192_local/x0pred/wandb/
offline-run-20260527_141814-86a3glc2
offline-run-20260527_142347-mtbkd4oj
offline-run-20260527_142651-wo1btysn
offline-run-20260527_165615-se7uhvo1
offline-run-20260528_010224-lw3dzxr5
... (15 separate runs total, through 20260612_220436)
```

15 fragmented runs, each only covering a slice of the 0→400,000 step range, with no single continuous loss curve. **This is purely a logging artifact** — it does not affect the checkpoint files in `ckpts/`, which are saved/loaded independently via `train_state.save()` / `train_state.resume()` ([utils.py:96-124](utils.py#L96-L124)).

### Step 1 — Archive the old fragmented runs (don't delete)

```bash
cd /mnt/zone_a/PQDiff/PQDiff
mv workdir/intel192_local/x0pred/wandb workdir/intel192_local/x0pred/wandb_archive_$(date +%Y%m%d)
```

This clears the slate without destroying the old offline logs, in case you want to `wandb sync` them later for historical reference.

### Step 2 — Pin a fixed run ID so future restarts continue the SAME run

`wandb.init()` reads the `WANDB_RUN_ID` and `WANDB_RESUME` environment variables automatically — no code change needed. Export these **once per shell session**, before every future launch of `train_ldm.py`:

```bash
export WANDB_RUN_ID=intel192_local_x0pred_v2
export WANDB_RESUME=allow
```

Put this in a small wrapper script (e.g. `train_resume.sh`) so you don't have to retype it:

```bash
#!/bin/bash
export WANDB_RUN_ID=intel192_local_x0pred_v2
export WANDB_RESUME=allow
cd /mnt/zone_a/PQDiff/PQDiff
source .venv/bin/activate
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
    train_ldm.py --config=configs/intel192_local.py
```

As long as `WANDB_RUN_ID` stays the same across every future restart, all subsequent offline runs will sync into **one continuous run** on the W&B dashboard instead of fragmenting again.

### Step 3 — Launch training

Run the wrapper script (or the plain command with the env vars exported). The first launch after Step 1 creates a fresh `wandb/offline-run-...` directory — this is the new, clean record.

### Step 4 — Sync to W&B cloud (when ready)

```bash
wandb sync workdir/intel192_local/x0pred/wandb/offline-run-*/
```

Since all future runs share `WANDB_RUN_ID=intel192_local_x0pred_v2`, syncing multiple offline runs appends to the same cloud run rather than creating new ones.

---

## 10. Sampling / Evaluation

After training, generate outpainted images. Note `evaluate.py` uses `torch.distributed.launch` with
`--nproc_per_node=2` in the original repo examples — given [§8a](#8a-why-multi_gpu-oom-on-these-cards)'s
finding that this model OOMs across 2 GPUs during *training* (optimizer state + DDP overhead), test with
`--nproc_per_node=1` first if you hit OOM here too; evaluation has no optimizer state so it may fit, but
hasn't been verified on this hardware yet.

```bash
python3 -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port=46123 \
    evaluate.py \
    --target_expansion 0.25 0.25 0.25 0.25 \
    --eval_dir ./eval_dir/scenery/1x/ \
    --size 128 \
    --config flickr192_local
```

`target_expansion` specifies padding fractions as `(top, bottom, left, right)`.

Compute metrics:

```bash
# Inception Score
python eval_dir/inception.py --path ./eval_dir/scenery/1x/gen/

# FID
python -m pytorch_fid ./eval_dir/scenery/1x/ori/ ./eval_dir/scenery/1x/gen/

# Centered PSNR
python eval_dir/psnr.py --original ./eval_dir/scenery/1x/ori/ \
                         --contrast ./eval_dir/scenery/1x/gen/
```

---

## 11. Data Flow Summary

```
Image (192×192×3)
    │
    ├─ Anchor crop (15–40%)  →  resize to 192×192  →  VAE encode  →  z_anchor (4×24×24)
    │                                               →  relative pos embedding (144×1024)
    │
    └─ Target crop (80–100%) →  resize to 192×192  →  VAE encode  →  z_target (4×24×24)

Training:
  z_target + noise ε  →  z_noisy
  UViT(z_noisy, [z_anchor, pos_query], t)  →  ε_pred
  Loss = ‖ε − ε_pred‖²

Inference (50-step DPM-Solver):
  z_T ~ N(0, I)  →  UViT iterative denoising with CFG  →  z_0  →  VAE decode  →  outpainted image
```
