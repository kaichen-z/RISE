"""Real-scene inference: two-stage feed-forward DDIM sampling on the maps from 4_inversion.

    python custom_eval.py --flagfile configs/eval_ddim.txt

Stage 1 samples the object map from the radar observation; stage 2 samples the room
layout conditioned on (observation, stage-1 objects).  Predictions are written to
`--eval_out/{obs,wall}/ourdiff_<case>_<ratio>.png` and scored by ../evaluate.py.

Feed-forward only: the latent-noise ("reverse") optimisation of the research code is
not part of this release -- see ../../README.md.

Against the `github` release this file
  * batches the evaluation set (`--batch_size`, was hard-coded to 1) and no longer
    drops the ragged tail (`drop_last=True` silently skipped the last case),
  * draws a fresh x_T per batch on the target device instead of reusing one CPU
    tensor for every case,
  * runs under autocast on CUDA,
  * loads the two stages as two distinct UNets (stage 2 takes a second conditioning
    image; feeding cond2 to a stage-1 model silently drops it).
"""

import os
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from absl import app, flags
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import common_flags
from common_flags import (FLAGS, amp_dtype, build_unet, get_device, load_ema, seed_everything,
                          tune_backend)
from data.data import CustomDataset
from models.diffusion import DDIMSampler

# Real-scene evaluation inputs (written by 4_inversion --step layout --export)
flags.DEFINE_string('path_obs', '../4_inversion/result_ours/result_input/image_lowr',
                    help='low-resolution observation maps')
flags.DEFINE_string('path_traj', '../4_inversion/result_ours/result_input/image_traj',
                    help='trajectory maps (same file names as --path_obs)')
flags.DEFINE_string('path_gt_obs', './data/GT/obs', help='object ground truth')
flags.DEFINE_string('path_gt_wall', './data/GT/wall', help='layout ground truth')
flags.DEFINE_string('eval_out', './eval_output', help='where predictions are written')
flags.DEFINE_string('ckpt_s1', './logs/stage1_object/last_ckpt.pt', help='stage-1 checkpoint')
flags.DEFINE_string('ckpt_s2', './logs/stage2_layout/last_ckpt.pt', help='stage-2 checkpoint')


def build_samplers(device):
    """Load both stages, each with its own UNet and sampler."""
    model_s1 = build_unet(image_cond1=True, image_cond2=False).to(device)
    model_s2 = build_unet(image_cond1=True, image_cond2=True).to(device)
    model_s1.load_state_dict(load_ema(FLAGS.ckpt_s1, device))
    model_s2.load_state_dict(load_ema(FLAGS.ckpt_s2, device))

    def wrap(model):
        model.eval().requires_grad_(False)
        sampler = DDIMSampler(model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
                              FLAGS.mean_type, eta=FLAGS.ddim_eta,
                              ddim_steps=FLAGS.ddim_steps).to(device)
        return torch.nn.DataParallel(sampler) if FLAGS.parallel else sampler

    return wrap(model_s1), wrap(model_s2)


def save_batch(pred, out_dir, cases, ratios):
    for i in range(pred.shape[0]):
        array = (pred[i, 0].float().cpu().numpy() * 255).astype(np.uint8)
        name = f"ourdiff_{int(cases[i])}_{int(ratios[i])}.png"
        Image.fromarray(array).save(os.path.join(out_dir, name))   # uint8 2-D -> mode "L"


def evaluate():
    device = get_device()
    tune_backend(device)
    seed_everything()
    dtype = amp_dtype(device)

    dataset = CustomDataset(FLAGS.path_obs, FLAGS.path_traj, FLAGS.path_gt_obs, FLAGS.path_gt_wall)
    loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False,
                        num_workers=FLAGS.num_workers, drop_last=False)
    sampler_s1, sampler_s2 = build_samplers(device)

    save_obs = os.path.join(FLAGS.eval_out, "obs")
    save_wall = os.path.join(FLAGS.eval_out, "wall")
    os.makedirs(save_obs, exist_ok=True)
    os.makedirs(save_wall, exist_ok=True)
    print(f"{len(dataset)} maps | {FLAGS.ddim_steps} DDIM steps of T={FLAGS.T} | device {device}")

    for image_obs, _traj, _gt_obs, _gt_wall, cases, ratios in tqdm(loader, dynamic_ncols=True):
        image_obs = image_obs.to(device)
        x_T = torch.randn(image_obs.shape[0], 1, FLAGS.img_size, FLAGS.img_size, device=device)
        ctx = torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()
        with torch.no_grad(), ctx:
            pred_obs = sampler_s1(x_T, image_obs).float()
            # stage 2 sees a hard object map, exactly as at training time (+-1 valued)
            second_input = torch.where(pred_obs > 0, 1., -1.).to(image_obs.dtype)
            pred_wall = sampler_s2(x_T, image_obs, second_input).float()
        save_batch((pred_obs > 0).float(), save_obs, cases, ratios)
        save_batch((pred_wall > 0).float(), save_wall, cases, ratios)


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    evaluate()


if __name__ == '__main__':
    app.run(main)
