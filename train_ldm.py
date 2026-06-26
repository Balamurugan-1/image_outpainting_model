import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from dataset.dataset import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    # num_processes must be known before constructing Accelerator (to size gradient_accumulation_steps),
    # so read it the same way accelerate itself does internally rather than constructing twice.
    num_processes = int(os.environ.get('WORLD_SIZE', 1))
    micro_batch_size = config.train.get('micro_batch_size', config.train.batch_size // num_processes)
    assert config.train.batch_size % (micro_batch_size * num_processes) == 0, \
        'train.batch_size must be an exact multiple of train.micro_batch_size * num_processes'
    accum_steps = config.train.batch_size // (micro_batch_size * num_processes)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=accum_steps)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')
    if accum_steps > 1:
        logging.info(f'Gradient accumulation: micro_batch_size={micro_batch_size} x accum_steps={accum_steps} '
                      f'x num_processes={num_processes} = effective batch_size={config.train.batch_size}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    mini_batch_size = micro_batch_size

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    # assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    if train_state.nnet_ema is not None:
        nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
            train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    else:
        nnet, optimizer, train_dataset_loader = accelerator.prepare(
            train_state.nnet, train_state.optimizer, train_dataset_loader)
        nnet_ema = None
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()


    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    # Falls back to the raw (non-EMA) model for eval/sampling previews when use_ema=False.
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE()) if nnet_ema is not None else score_model


    def train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target):

        _metrics = dict()
        with accelerator.accumulate(nnet):
            if config.train.mode == 'uncond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred)
            elif config.train.mode == 'cond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
                loss = sde.LSimple(score_model, _z, pred=config.pred, conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError(config.train.mode)
            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            if not torch.isfinite(_metrics['loss']):
                logging.warning(f'step {train_state.step}: non-finite loss ({_metrics["loss"].item()}), skipping micro-batch')
                optimizer.zero_grad()
                return dict(lr=train_state.optimizer.param_groups[0]['lr'], stepped=False, **_metrics)
            accelerator.backward(loss.mean())
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.get('grad_clip', 1.0))
            optimizer.step()
            if accelerator.sync_gradients:
                lr_scheduler.step()
            optimizer.zero_grad()
        stepped = accelerator.sync_gradients
        if stepped:
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], stepped=stepped, **_metrics)

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        batch = [batch[i].float() for i in range(len(batch))]
        prime_target, prime_anchor_view, prime_targe_pos = batch
        encode_anchor, encode_target = encode(prime_anchor_view), encode(prime_target)
        metrics = train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target)
        if not metrics.pop('stepped'):
            continue  # still accumulating gradients, no effective step completed yet

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 1:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            z_init = torch.randn(encode_target.size(), device=device)
            if config.train.mode == 'uncond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50)
            elif config.train.mode == 'cond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50, conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError

            # through diffusion
            pred_target = decode(z)
            pred_target = make_grid(dataset.unpreprocess(pred_target), 10)

            # through autoencoder
            decode_target = decode(encode_target)
            decode_target = make_grid(dataset.unpreprocess(decode_target), 10)

            # ground truth
            prime_target = make_grid(dataset.unpreprocess(prime_target), 10)

            # through autoencoder
            decode_anchor = decode(encode_anchor)
            decode_anchor = make_grid(dataset.unpreprocess(decode_anchor), 10)

            # prime condition
            prime_anchor_view = make_grid(dataset.unpreprocess(prime_anchor_view), 10)

            save_image(pred_target, os.path.join(config.sample_dir, f'predict_target-{train_state.step}.png'))
            save_image(decode_target, os.path.join(config.sample_dir, f'decode_target-{train_state.step}.png'))
            save_image(prime_target, os.path.join(config.sample_dir, f'prime_target-{train_state.step}.png'))
            save_image(decode_anchor, os.path.join(config.sample_dir, f'decode_anchor-{train_state.step}.png'))
            save_image(prime_anchor_view, os.path.join(config.sample_dir, f'prime_anchor-{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(pred_target)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    accelerator.wait_for_everyone()


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'x0pred'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
