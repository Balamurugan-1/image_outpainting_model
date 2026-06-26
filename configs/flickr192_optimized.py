import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 24, 24)

    config.autoencoder = d(
        pretrained_path='assets/stable_diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=50000,
        # Effective batch = compromise between batch=4 (flickr192_local, too noisy to converge cleanly —
        # see improvement.md root cause #2) and a literal batch=256 (flickr192_large, ~6 min/step, 80,000
        # steps would take ~333 days on this GPU). batch=16 = 4x less gradient noise than batch=4 at 8x
        # the per-step time of batch=4, keeping a full run feasible in days. Raise toward 64 (closer to
        # the paper) once GPU headroom allows a larger micro_batch_size — see improvement.md §5.4/5.5.
        batch_size=16,
        # actual per-step GPU batch, accumulated 8x to reach batch_size=16. NOTE (2026-06-24): tested
        # micro_batch_size=1, 2, and 4 with use_ema=True — all three needed ~4.1GB and OOM'd on the very
        # FIRST micro-batch, before optimizer.step() ever ran (accum_steps>1 delays the first real step).
        # That means the crash predates any optimizer-state allocation — 8-bit Adam can't fix it; the
        # floor there was raw UViT + EMA UViT + gradient buffers + VAE + CUDA context, none of which
        # scale with micro_batch_size. use_ema=False below removes the EMA copy (~1.16GB) from that floor.
        micro_batch_size=2,
        use_ema=False,  # see improvement.md §5.6 — frees ~1.16GB; re-enable once more GPU headroom exists
        mode='cond',
        log_interval=10,
        eval_interval=500,
        save_interval=500,  # was 10000 in flickr192_large — that gap meant 0 checkpoints across 19 hours
    )

    config.optimizer = d(
        # 8-bit AdamW (bitsandbytes): int8 momentum buffers instead of fp32, ~halving optimizer-state
        # memory (2.32GB -> ~1.16GB for this model). This — not micro_batch_size — was the lever that
        # actually freed enough headroom to fit alongside the other job on GPU1. See improvement.md §5.6.
        name='adamw8bit',
        # Linear-scaled from the paper's lr=2e-4 at batch=256 (Goyal et al. linear scaling rule):
        # 2e-4 * (64/256) = 5e-5. See improvement.md root cause #2.
        lr=0.00005,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        # 'customized' (used by both flickr192_local and flickr192_large) holds lr constant forever after
        # warmup — improvement.md root cause #1, the main driver of the observed loss plateau. This decays.
        name='cosine_warmup',
        warmup_steps=2000,
        total_steps=30000,   # cosine reaches min_lr_ratio by this step; n_steps=50000 leaves margin beyond it
        min_lr_ratio=0.01,   # anneal down to 1% of peak lr (5e-7), not all the way to 0
    )

    config.nnet = d(
        # Unchanged from flickr192_local/flickr192_large — architecture changes come in a later pass.
        name='uvit',
        img_size=24,
        patch_size=2,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=True
    )

    config.dataset = d(
        name='flickr',
        path='./dataset/scenery/all/',
        resolution=192,
        embed_dim=1024,
        grid_size=12,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=1,
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config
