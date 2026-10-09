"""Train one stage of the SRHD hierarchical diffusion model (feed-forward only).

    python main.py --flagfile configs/nghost_diff1.txt     # stage 1: objects
    python main.py --flagfile configs/nghost_diff2.txt     # stage 2: layout

Stage 1 learns p(gt_obs | observation); stage 2 learns p(gt_wall | observation,
gt_obs).  Training data are the augmented simulation bundles from 6_simulator.

What changed against the `github` release (same model, same loss, same schedule):
  * mixed precision (bf16/fp16 autocast + GradScaler) and TF32 matmuls,
  * fused Adam, foreach EMA (one kernel per dtype instead of a Python loop over
    every tensor of the state dict, at every step),
  * the training set is cached bit-packed in RAM and the loader keeps its workers
    alive between epochs,
  * previews are DDIM (20 steps) instead of the 500-step ancestral chain,
  * `last_ckpt.pt` + `--resume`, and only `--keep_last` periodic checkpoints are
    kept (a full run wrote 40 x 1.1 GB),
  * `--accum_steps` so a small GPU can still reach the paper's effective batch,
  * evaluation runs batched and on the right device (the old `eval` built x_T on
    the CPU, which simply crashed on CUDA).
"""

import contextlib
import copy
import os
import time
import warnings

import torch
from absl import app
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange

import common_flags
from common_flags import (FLAGS, amp_dtype, build_unet, get_device, load_ema, maybe_compile,
                          seed_everything, tune_backend)
from data.data import NumpyDataset
from models.diffusion import DDIMSampler, GaussianDiffusionTrainer
from models.metrics import obs_metrics_torch


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def autocast(device, dtype):
    """autocast when a mixed dtype was resolved, a no-op context otherwise."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


@torch.no_grad()
def ema_update(source, target, decay):
    """EMA over parameters *and* buffers, batched with the _foreach_ kernels."""
    src_params, tgt_params = list(source.parameters()), list(target.parameters())
    torch._foreach_mul_(tgt_params, decay)
    torch._foreach_add_(tgt_params, src_params, alpha=1. - decay)
    for src_buf, tgt_buf in zip(source.buffers(), target.buffers()):
        tgt_buf.copy_(src_buf)


def build_loader(path, batch_size, shuffle, drop_last):
    dataset = NumpyDataset(path, FLAGS.H, FLAGS.W, cache=FLAGS.cache_data)
    workers = FLAGS.num_workers if not FLAGS.cache_data else 0   # cached data needs no workers
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0), dataset


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def rotate_checkpoints(logdir, keep_last):
    if keep_last is None or keep_last <= 0:
        return
    ckpts = sorted((f for f in os.listdir(logdir) if f.endswith('_ckpt.pt') and f[0].isdigit()),
                   key=lambda f: int(f.split('_')[0]))
    for stale in ckpts[:-keep_last]:
        os.remove(os.path.join(logdir, stale))


def train():
    device = get_device()
    tune_backend(device)
    seed_everything()
    stage2 = FLAGS.image_cond1 and FLAGS.image_cond2
    dtype = amp_dtype(device)
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)

    loader, dataset = build_loader(FLAGS.train_data, FLAGS.batch_size, shuffle=True, drop_last=True)
    looper = infinite(loader)

    net_model = build_unet(FLAGS.image_cond1, FLAGS.image_cond2).to(device)
    if FLAGS.channels_last and device.type == 'cuda':
        net_model = net_model.to(memory_format=torch.channels_last)
    ema_model = copy.deepcopy(net_model).eval().requires_grad_(False)

    try:
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr, fused=(device.type == 'cuda'))
    except (TypeError, RuntimeError):       # fused Adam needs a recent PyTorch + CUDA tensors
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    start_step = 0
    resume_path = os.path.join(FLAGS.logdir, 'last_ckpt.pt')
    if FLAGS.resume and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        if ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step']
        print(f"resumed from {resume_path} at step {start_step}")
    else:
        print("no checkpoint found, training from scratch")

    trainer = GaussianDiffusionTrainer(net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    ema_sampler = DDIMSampler(ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                              FLAGS.mean_type, eta=FLAGS.ddim_eta, ddim_steps=FLAGS.ddim_steps).to(device)
    trainer.model = maybe_compile(trainer.model, device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as handle:
        handle.write(FLAGS.flags_into_string())
    n_params = sum(p.numel() for p in net_model.parameters())
    print(f"stage {'2 (layout)' if stage2 else '1 (objects)'} | {len(dataset)} bundles | "
          f"{n_params / 1e6:.1f} M params | device {device} | "
          f"precision {'fp32' if dtype is None else str(dtype).split('.')[-1]}")

    x_T_preview = torch.randn(FLAGS.sample_size, 1, FLAGS.img_size, FLAGS.img_size, device=device)
    loss_log = open(os.path.join(FLAGS.logdir, 'loss.csv'), 'a')
    running, t_start = None, time.time()

    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True, initial=start_step,
                total=FLAGS.total_steps) as pbar:
        for step in pbar:
            optim.zero_grad(set_to_none=True)
            for _ in range(FLAGS.accum_steps):
                observation, gt_obs, gt_wall = (t.to(device, non_blocking=True) for t in next(looper))
                with autocast(device, dtype):
                    if stage2:
                        loss = trainer(gt_wall, observation, gt_obs).mean()
                    else:
                        loss = trainer(gt_obs, observation).mean()
                scaler.scale(loss / FLAGS.accum_steps).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step()
            ema_update(net_model, ema_model, FLAGS.ema_decay)

            running = loss.item() if running is None else 0.99 * running + 0.01 * loss.item()
            pbar.set_postfix(loss='%.4f' % running)
            if step % 100 == 0:
                loss_log.write(f"{step},{running:.6f},{time.time() - t_start:.1f}\n")
                loss_log.flush()

            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                with torch.no_grad(), autocast(device, dtype):
                    cond = observation[:FLAGS.sample_size]
                    cond2 = gt_obs[:FLAGS.sample_size] if stage2 else None
                    x_T = x_T_preview[:cond.shape[0]]
                    x_0 = ema_sampler(x_T, cond, cond2) if stage2 else ema_sampler(x_T, cond)
                target = gt_wall if stage2 else gt_obs
                for name, tensor in (('observation', cond), ('gt', target[:cond.shape[0]]),
                                     ('prediction', x_0.float())):
                    grid = (make_grid(tensor) + 1) / 2
                    save_image(grid, os.path.join(FLAGS.logdir, 'sample', f'{step}_{name}.png'))

            if FLAGS.save_step > 0 and step > 0 and step % FLAGS.save_step == 0:
                ckpt = {'net_model': net_model.state_dict(), 'ema_model': ema_model.state_dict(),
                        'sched': sched.state_dict(), 'optim': optim.state_dict(),
                        'scaler': scaler.state_dict(), 'step': step}
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'{step}_ckpt.pt'))
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'last_ckpt.pt'))
                rotate_checkpoints(FLAGS.logdir, FLAGS.keep_last)

    ckpt = {'net_model': net_model.state_dict(), 'ema_model': ema_model.state_dict(),
            'sched': sched.state_dict(), 'optim': optim.state_dict(),
            'scaler': scaler.state_dict(), 'step': FLAGS.total_steps}
    torch.save(ckpt, os.path.join(FLAGS.logdir, f'{FLAGS.total_steps}_ckpt.pt'))
    torch.save(ckpt, os.path.join(FLAGS.logdir, 'last_ckpt.pt'))
    loss_log.close()


def eval_ddim():
    """Score a trained stage on held-out *simulated* data (IoU / Dice)."""
    device = get_device()
    tune_backend(device)
    seed_everything()
    stage2 = FLAGS.image_cond1 and FLAGS.image_cond2
    dtype = amp_dtype(device)
    loader, dataset = build_loader(FLAGS.test_data, FLAGS.batch_size, shuffle=False, drop_last=False)

    model = build_unet(FLAGS.image_cond1, FLAGS.image_cond2).to(device)
    ckpt_path = FLAGS.ckpt if os.path.isabs(FLAGS.ckpt) else os.path.join(FLAGS.logdir, FLAGS.ckpt)
    model.load_state_dict(load_ema(ckpt_path, device))
    model.eval()
    sampler = DDIMSampler(model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                          FLAGS.mean_type, eta=FLAGS.ddim_eta, ddim_steps=FLAGS.ddim_steps).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    print(f"evaluating {ckpt_path} on {len(dataset)} held-out bundles")

    iou_sum = dice_sum = count = 0
    for observation, gt_obs, gt_wall in tqdm(loader, dynamic_ncols=True):
        observation, gt_obs, gt_wall = (t.to(device) for t in (observation, gt_obs, gt_wall))
        x_T = torch.randn(observation.shape[0], 1, FLAGS.img_size, FLAGS.img_size, device=device)
        with autocast(device, dtype):
            pred = sampler(x_T, observation, gt_obs) if stage2 else sampler(x_T, observation)
        target = gt_wall if stage2 else gt_obs
        # per-sample, then averaged: obs_metrics_torch flattens whatever it is given, so
        # feeding it a whole batch would micro-average pixels instead of averaging cases.
        for i in range(observation.shape[0]):
            iou, dice = obs_metrics_torch(prediction=(pred[i:i + 1].float() > 0).float(),
                                          ground_truth=target[i:i + 1])
            iou_sum += float(iou)
            dice_sum += float(dice)
            count += 1
    print(f"----- IoU  : {iou_sum / max(count, 1):.4f}")
    print(f"----- Dice : {dice_sum / max(count, 1):.4f}")


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train_ddpm:
        print('===== training =====')
        train()
    if FLAGS.eval_ddim:
        print('===== evaluation (simulated) =====')
        eval_ddim()
    if not (FLAGS.train_ddpm or FLAGS.eval_ddim):
        print("nothing to do: pass --train_ddpm or --eval_ddim")


if __name__ == '__main__':
    app.run(main)
