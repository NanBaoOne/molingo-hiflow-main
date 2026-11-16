import sys
sys.path.append('.')

import os
import argparse
import torch
import math
import copy
import numpy as np

from os.path import join as pjoin
from torch.utils.data import DataLoader
from argparse import Namespace
from collections import OrderedDict

import mogen.models.molingo.molingo as molingo_models
import mogen.models.hiflow.hiflow_sae as hiflow_models

from mogen.utils.fixseed import fixseed
from mogen.utils.get_opt import get_opt
from mogen.utils.eval_utils import load_ms_evaluators

from mogen.data.ms_dataset import Text2MotionDatasetMS
from mogen.models.vae.vae import VAE
from mogen.core.eval import eval_molingo_ms

def load_vae_model(vae_opt, ckpt_path, dim_pose, device):
    vae_model = VAE(
        input_width=dim_pose,
        output_emb_width=vae_opt.output_emb_width,
        down_t=vae_opt.down_t,
        stride_t=vae_opt.stride_t,
        width=vae_opt.width,
        depth=vae_opt.depth,
        dilation_growth_rate=vae_opt.dilation_growth_rate,
        activation=vae_opt.activation,
        norm=vae_opt.norm,
        pad_mode=vae_opt.pad_mode,
        ae=vae_opt.ae,
    )
    vae_model.to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[len("module."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    vae_model.load_state_dict(state_dict=new_state_dict)
    for param in vae_model.parameters():
        param.requires_grad = False
    return vae_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurations for Babel Model Eval")
    parser.add_argument("-d", "--dim_pose", type=int, default=272, choices=[263, 272], help="Motion representation")
    parser.add_argument("-s", "--step", type=int, default=32, help="Sampling steps")
    parser.add_argument("-c", "--cfg", type=float, default=5.5, help="CFG value")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("-a", "--acc", type=int, default=3, help="Sampling reduction factor")
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Split for evaluation (Babel has no test set)")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='second-stage checkpoint root or model subdirectory')

    args = parser.parse_args()

    fixseed(3407)

    device = 'cuda'
    dim_pose = args.dim_pose
    step = args.step
    cfg = args.cfg
    acc = args.acc
    repeat_times = args.repeat

    opt = Namespace()
    opt.dataset_name = 'babel' if dim_pose == 272 else 't2m'

    # create eval output file
    requested_model_dir = args.model_dir or pjoin('mogen/checkpoints', opt.dataset_name, f'pretrained_model_{dim_pose}')
    if os.path.exists(pjoin(requested_model_dir, 'opt.txt')):
        run_dir = requested_model_dir
        ckpt_dir = pjoin(run_dir, 'model')
    elif os.path.basename(os.path.normpath(requested_model_dir)) == 'model' and os.path.exists(pjoin(os.path.dirname(requested_model_dir), 'opt.txt')):
        run_dir = os.path.dirname(requested_model_dir)
        ckpt_dir = requested_model_dir
    else:
        run_dir = requested_model_dir
        ckpt_dir = pjoin(run_dir, 'model')
    opt_path = pjoin(run_dir, 'opt.txt')
    model_opt = get_opt(opt_path, device)

    eval_output_dir = pjoin(run_dir, f'eval_babel_cfg_{cfg}_step_{step}_acc_{acc}_{args.split}')
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_file = pjoin(eval_output_dir, 'eval_res.txt')

    # Babel data path
    opt.joints_num = 22
    opt.data_root = '/mnt/user/lh3/MotionStreamer-main/babel_272'
    opt.motion_dir = pjoin(opt.data_root, 'motion_data')
    opt.text_dir = pjoin(opt.data_root, 'texts')

    # Babel evaluators
    textencoder, motionencoder = load_ms_evaluators(device=device)

    # load vae
    if not hasattr(model_opt, 'sae_name'):
        model_opt.sae_name = model_opt.vae
    if not hasattr(model_opt, 'sae_ckpt_name'):
        model_opt.sae_ckpt_name = 'net_best_fid.ckpt'
    checkpoints_dir = getattr(model_opt, 'checkpoints_dir', 'mogen/checkpoints')
    vae_dataset = getattr(model_opt, 'vae_dataset_name', None) or opt.dataset_name
    vae_opt_path = pjoin(checkpoints_dir, vae_dataset, model_opt.sae_name, 'opt.txt')
    vae_ckpt_path = pjoin(checkpoints_dir, vae_dataset, model_opt.sae_name, 'model', model_opt.sae_ckpt_name)
    vae_opt = get_opt(vae_opt_path, device=device)
    vae_model = load_vae_model(vae_opt, vae_ckpt_path, dim_pose, device=device)

    vae_embed_dim = vae_opt.output_emb_width
    ds_rate = math.pow(2, vae_opt.down_t)
    ds_rate = int(ds_rate)
    opt.unit_length = ds_rate

    # Babel test/eval dataset (no test split, use val)
    mean = np.load(pjoin(opt.data_root, 't2m_babel_mean_std', 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 't2m_babel_mean_std', 'Std.npy'))
    test_split_file = pjoin(opt.data_root, 'split', f'{args.split}.txt')
    opt.max_motion_length = 300
    test_dataset = Text2MotionDatasetMS(opt, mean, std, test_split_file)
    test_loader = DataLoader(test_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)

    # second-stage model initialization
    second_stage = getattr(model_opt, 'second_stage', 'molingo')
    stage_models = molingo_models if second_stage == 'molingo' else hiflow_models
    stage_prefix = 'molingo' if second_stage == 'molingo' else 'hiflow'
    model_func_name = f'{stage_prefix}_{model_opt.model_size}'
    molingo_func = getattr(stage_models, model_func_name)
    partial_molingo = molingo_func()
    adapter_layers = getattr(model_opt, 'adapter_layers', getattr(model_opt, 'aligner_layers', 6))

    model_kwargs = dict(vae_embed_dim=vae_embed_dim,
                        token_size=opt.max_motion_length // ds_rate,
                        unit_length=ds_rate,
                        sample_steps=step,
                        t5_max_len=model_opt.t5_max_len,
                        adapter_layers=adapter_layers,
                        ae=vae_opt.ae)
    if second_stage == 'hiflow':
        model_kwargs['time_patch'] = getattr(model_opt, 'time_patch', 2)
        model_kwargs['scales'] = getattr(model_opt, 'hiflow_scales', '0.3,0.6,1.0')
    molingo_model = partial_molingo(**model_kwargs)
    molingo_model.to(device)
    model_without_ddp = molingo_model

    # load molingo model
    checkpoint = torch.load(pjoin(ckpt_dir, f"net_best_fid.pth"), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    model_params = list(model_without_ddp.parameters())
    ema_state_dict = checkpoint['model_ema']
    ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
    del checkpoint
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = ema_params[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)
    del ema_state_dict
    model_without_ddp.eval()

    # eval
    fid_ms, top1_ms, top2_ms, top3_ms, matching_score_ms, = [], [], [], [], []
    model_without_ddp.eval()

    with open(eval_file, 'w') as f:
        for rt in range(repeat_times):
            tmp_fid_ms, tmp_top1_ms, tmp_top2_ms, tmp_top3_ms, \
                tmp_matching_score_ms, = eval_molingo_ms(test_loader, model_without_ddp, vae_model,
                                                       ep=rt, cfg=cfg,
                                                       motionencoder=motionencoder, textencoder=textencoder,
                                                       std_factor=model_opt.std_factor, acc_ratio=acc)

            fid_ms.append(tmp_fid_ms)
            top1_ms.append(tmp_top1_ms)
            top2_ms.append(tmp_top2_ms)
            top3_ms.append(tmp_top3_ms)
            matching_score_ms.append(tmp_matching_score_ms)

        fid_ms = np.array(fid_ms)
        top1_ms = np.array(top1_ms)
        top2_ms = np.array(top2_ms)
        top3_ms = np.array(top3_ms)
        matching_score_ms = np.array(matching_score_ms)

        msg_final_ms = f"\tTMR FID: {np.mean(fid_ms):.3f}, conf. {np.std(fid_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                        f"\tTOP1: {np.mean(top1_ms):.3f}, conf. {np.std(top1_ms) * 1.96 / np.sqrt(repeat_times):.3f}, TOP2. {np.mean(top2_ms):.3f}, conf. {np.std(top2_ms) * 1.96 / np.sqrt(repeat_times):.3f}, TOP3. {np.mean(top3_ms):.3f}, conf. {np.std(top3_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n" \
                        f"\tMatching Score: {np.mean(matching_score_ms):.3f}, conf. {np.std(matching_score_ms) * 1.96 / np.sqrt(repeat_times):.3f}\n"
        print(msg_final_ms)
        print(msg_final_ms, file=f, flush=True)

    print('Done calculating metrics')
