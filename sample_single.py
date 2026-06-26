import argparse
import importlib
import os

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

import libs.autoencoder
import sde
import utils
from dataset.pos import get_2d_local_sincos_pos_embed
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


def calculate_sin_cos(lpos, gpos, embed_dim=1024, grid_size=12):
    kg = gpos[3] / grid_size
    w_bias = (lpos[1] - gpos[1]) / kg
    kl = lpos[3] / grid_size
    w_scale = kl / kg
    kg = gpos[2] / grid_size
    h_bias = (lpos[0] - gpos[0]) / kg
    kl = lpos[2] / grid_size
    h_scale = kl / kg
    return get_2d_local_sincos_pos_embed(embed_dim, grid_size, w_bias, w_scale, h_bias, h_scale)


def calculate_input_pos(target):
    init_location = (1000, 1000, 256, 256)
    top, down, left, right = target
    i = init_location[0] - int(256 * top)
    j = init_location[1] - int(256 * left)
    h = int(256 * (top + down)) + 256
    w = int(256 * (left + right)) + 256
    return init_location, (i, j, h, w)


def preprocess_image(image_path, crop_size):
    tfm_anchor = transforms.Compose([
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])
    tfm_target = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    anchor = tfm_anchor(image) * 2.0 - 1.0
    target = tfm_target(image) * 2.0 - 1.0
    return anchor.unsqueeze(0), target.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser("Sample a single image with a trained PQDiff outpainting model")
    parser.add_argument("--image", required=True, type=str, help="Path to a sample image.")
    parser.add_argument("--config", default="flickr192_local", type=str, help="Config module name without .py")
    parser.add_argument("--ckpt_step", default=None, type=int, help="Checkpoint step to load. Defaults to latest.")
    parser.add_argument("--target_expansion", nargs=4, type=float, default=(0.25, 0.25, 0.25, 0.25))
    parser.add_argument("--crop_size", type=int, default=128, help="Center crop size for the visible anchor.")
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--output", default="sample_outputs", type=str)
    args = parser.parse_args()

    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.get_config()
    config.config_name = args.config
    config.hparams = "x0pred"
    config.workdir = os.path.join("workdir", config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, "ckpts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path).to(device)
    train_state = utils.initialize_train_state(config, device)
    train_state.resume(config.ckpt_root, step=args.ckpt_step)
    train_state.nnet_ema.eval()

    score_model = sde.ScoreModel(train_state.nnet_ema, pred=config.pred, sde=sde.VPSDE())
    anchor_pos, target_pos = calculate_input_pos(args.target_expansion)
    prime_target_pos = torch.FloatTensor(
        calculate_sin_cos(target_pos, anchor_pos, embed_dim=config.dataset.embed_dim, grid_size=config.dataset.grid_size)
    ).unsqueeze(0).to(device)

    anchor_img, target_img = preprocess_image(args.image, args.crop_size)
    anchor_img = anchor_img.to(device)
    target_img = target_img.to(device)

    with torch.no_grad():
        encode_anchor = autoencoder.encode(anchor_img)
        z_init = torch.randn_like(encode_anchor)
        noise_schedule = NoiseScheduleVP(schedule="linear")
        kwargs = {"conditions": [encode_anchor, prime_target_pos.float()]}
        model_fn = model_wrapper(score_model.noise_pred, noise_schedule, time_input_type="0", model_kwargs=kwargs)
        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        z = dpm_solver.sample(
            z_init, steps=args.sample_steps, eps=1e-4, adaptive_step_size=False, fast_version=False
        )
        pred_target = autoencoder.decode(z)

    anchor_vis = (anchor_img.cpu() + 1.0) * 0.5
    target_vis = (target_img.cpu() + 1.0) * 0.5
    pred_vis = (pred_target.cpu() + 1.0) * 0.5

    save_image(anchor_vis, os.path.join(args.output, "anchor.png"))
    save_image(target_vis, os.path.join(args.output, "target_resized.png"))
    save_image(pred_vis, os.path.join(args.output, "generated.png"))
    comparison = torch.cat([anchor_vis, pred_vis, target_vis], dim=0)
    save_image(comparison, os.path.join(args.output, "comparison.png"), nrow=3)

    print(f"Saved outputs to {args.output}")


if __name__ == "__main__":
    main()
