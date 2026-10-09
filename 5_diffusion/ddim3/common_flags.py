"""Flags shared by `main.py` (training) and `custom_eval.py` (real-scene inference).

The `github` release declared the same ~40 absl flags twice, in two files that had
drifted apart (and still carried CIFAR-10 / FID leftovers from the upstream DDPM
repo).  They are declared once here instead; the config files under `configs/`
are unchanged apart from the dropped dead entries.
"""

import os

import torch
from absl import flags

FLAGS = flags.FLAGS

# ---------------------------------------------------------------- UNet
flags.DEFINE_integer('_Type_', 1, help='which stage this config trains (1 objects, 2 layout)')
flags.DEFINE_integer('input_ch', 1, help='input channels of the UNet')
flags.DEFINE_integer('ch', 128, help='base channel width')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier per level')
flags.DEFINE_multi_integer('attn', [1], help='levels that get self-attention')
flags.DEFINE_integer('num_res_blocks', 2, help='resblocks per level')
flags.DEFINE_float('dropout', 0.1, help='resblock dropout')
flags.DEFINE_integer('img_size', 128, help='image size')
flags.DEFINE_integer('H', 128, help='input map height')
flags.DEFINE_integer('W', 128, help='input map width')
flags.DEFINE_bool('image_cond1', True, help='condition on the radar observation (stage 1 and 2)')
flags.DEFINE_bool('image_cond2', False, help='condition on the stage-1 object map (stage 2 only)')

# ---------------------------------------------------------------- diffusion
flags.DEFINE_float('beta_1', 1e-4, help='first beta')
flags.DEFINE_float('beta_T', 0.02, help='last beta')
flags.DEFINE_integer('T', 500, help='diffusion steps used at training time')
flags.DEFINE_enum('mean_type', 'epsilon', ['epsilon'], help='prediction target')
flags.DEFINE_float('ddim_eta', 0.0, help='DDIM stochasticity (0 = deterministic)')
flags.DEFINE_integer('ddim_steps', 20, help='DDIM steps used for sampling')

# ---------------------------------------------------------------- optimisation
flags.DEFINE_float('lr', 2e-4, help='peak learning rate')
flags.DEFINE_float('grad_clip', 1., help='gradient-norm clipping')
flags.DEFINE_integer('total_steps', 400000, help='training steps')
flags.DEFINE_integer('warmup', 5000, help='linear warmup steps')
flags.DEFINE_integer('batch_size', 24, help='batch size')
flags.DEFINE_integer('accum_steps', 1, help='gradient-accumulation steps (effective batch = batch_size * accum_steps)')
flags.DEFINE_integer('num_workers', 4, help='dataloader workers')
flags.DEFINE_float('ema_decay', 0.9999, help='EMA decay of the sampling weights')
flags.DEFINE_bool('parallel', False, help='wrap the model in DataParallel (multi-GPU, single node)')
flags.DEFINE_enum('precision', 'auto', ['auto', 'fp32', 'bf16', 'fp16'],
                  help='training/inference precision; auto = bf16 on Ampere+, fp16 on older CUDA, fp32 elsewhere')
flags.DEFINE_bool('compile', False, help='torch.compile the UNet (PyTorch >= 2.1, CUDA)')
flags.DEFINE_bool('channels_last', True, help='use channels-last memory format on CUDA')
flags.DEFINE_integer('seed', 0, help='random seed (<0 disables seeding)')

# ---------------------------------------------------------------- logging
flags.DEFINE_string('logdir', './logs/stage1_object', help='checkpoint / sample directory')
flags.DEFINE_integer('sample_size', 8, help='number of preview samples')
flags.DEFINE_integer('sample_step', 5000, help='preview sampling frequency, 0 to disable')
flags.DEFINE_integer('save_step', 10000, help='checkpoint frequency, 0 to disable')
flags.DEFINE_integer('keep_last', 3, help='how many periodic checkpoints to keep (<=0 keeps all)')
flags.DEFINE_string('ckpt', 'last_ckpt.pt', help='checkpoint file used by --eval_ddim')
flags.DEFINE_bool('resume', True, help='resume from <logdir>/last_ckpt.pt when present')

# ---------------------------------------------------------------- data
flags.DEFINE_string('train_data', './data/train', help='augmented training .npy bundles (6_simulator)')
flags.DEFINE_string('test_data', './data/test', help='held-out simulated .npy bundles')
flags.DEFINE_bool('cache_data', True, help='keep the (bit-packed) bundles in RAM')

# ---------------------------------------------------------------- modes
flags.DEFINE_bool('train_ddpm', False, help='train from scratch / resume')
flags.DEFINE_bool('eval_ddim', False, help='score the checkpoint on held-out simulated data')


def get_device():
    return torch.device(os.environ.get('RISE_DEVICE')
                        or ('cuda:0' if torch.cuda.is_available() else 'cpu'))


def amp_dtype(device):
    """Resolve --precision into an autocast dtype (None = run in fp32)."""
    choice = FLAGS.precision
    if choice == 'fp32':
        return None
    if choice == 'bf16':
        return torch.bfloat16
    if choice == 'fp16':
        return torch.float16
    if device.type == 'cuda':
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None            # CPU / MPS: autocast buys nothing here, keep fp32


def tune_backend(device):
    """Enable the free CUDA fast paths (TF32 matmuls, cudnn autotuning)."""
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def seed_everything():
    if FLAGS.seed is not None and FLAGS.seed >= 0:
        torch.manual_seed(FLAGS.seed)
        import numpy as np
        np.random.seed(FLAGS.seed)


def build_unet(image_cond1, image_cond2):
    from models.UNet import UNet
    return UNet(T=FLAGS.T, input_ch=FLAGS.input_ch, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult,
                attn=FLAGS.attn, num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
                image_cond1=image_cond1, image_cond2=image_cond2)


def maybe_compile(model, device):
    if FLAGS.compile and device.type == 'cuda' and hasattr(torch, 'compile'):
        return torch.compile(model)
    return model


def load_ema(path, device):
    """Read the EMA weights out of a checkpoint written by main.py.

    Accepts the `github` layout ('ema_model' / legacy 'ema_model_s2') and strips a
    DataParallel / torch.compile prefix if one is present.
    """
    ckpt = torch.load(path, map_location=device)
    for key in ('ema_model', 'ema_model_s2'):
        if key in ckpt:
            state = ckpt[key]
            break
    else:
        raise KeyError(f"{path}: no 'ema_model' / 'ema_model_s2' entry (found {sorted(ckpt)})")
    clean = {}
    for k, v in state.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean[k] = v
    return clean
