"""Use the repository's original 272-d evaluator classes and metric functions."""
import bootstrap  # noqa: F401
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from progress import tqdm
from data272 import Motion272


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_evaluator(checkpoint, text_model, device):
    from mogen.models.eval.ms_motionencoder import ActorAgnosticEncoder
    from mogen.models.eval.distillbert_actor import DistilbertActorAgnosticEncoder
    motion = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)
    text = DistilbertActorAgnosticEncoder(text_model, num_layers=4, latent_dim=256)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    for prefix, model in [('motionencoder.', motion), ('textencoder.', text)]:
        weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False).to(device)
    return text, motion


@torch.no_grad()
def evaluate(args, vq, mask=None, residual=None, *, split='val', repeat=0, num_res_layers=-1):
    from mogen.utils.metrics import (calculate_activation_statistics, calculate_frechet_distance,
                                     calculate_diversity)
    dataset = Motion272(args.data_root, split, training=False)
    # Fixed candidate pool (32), independently of training batch size/GPU tuning.
    generator = torch.Generator().manual_seed(args.seed + repeat)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, generator=generator)
    if len(loader) == 0:
        raise ValueError('Evaluation requires at least 32 eligible samples')
    text_encoder, motion_encoder = load_evaluator(args.evaluator, args.eval_text_model, args.device)
    vq.eval()
    if mask is not None:
        mask.eval()
    if residual is not None:
        residual.eval()
    real, generated = [], []
    correct, real_correct = np.zeros(3), np.zeros(3)
    matching, real_matching, count = 0., 0., 0
    for batch in tqdm(loader, desc=f'{split} TMR272 / repeat {repeat}', dynamic_ncols=True):
        x = batch['motion'].to(args.device, non_blocking=True)
        lens = batch['length'].to(args.device)
        captions = list(batch['caption'])
        if mask is None:
            # Evaluation uses every residual codebook without quantizer dropout.
            ids, _ = vq.encode(x)
            y = vq.forward_decoder(ids)
        else:
            ids = mask.generate(captions, lens // 4, timesteps=args.sample_steps,
                                cond_scale=args.cfg, temperature=args.temperature,
                                topk_filter_thres=0.9, gsample=True)
            if residual is None:
                ids = ids.unsqueeze(-1)
                valid = torch.arange(ids.shape[1], device=lens.device)[None, :] < (lens // 4)[:, None]
                ids = ids.masked_fill(~valid[..., None], -1)
            else:
                ids = residual.generate(ids, captions, lens // 4, temperature=args.temperature,
                                        cond_scale=args.res_cfg, num_res_layers=num_res_layers)
            y = vq.forward_decoder(ids)
        y = F.pad(y.float(), (0, 0, 0, 300 - y.shape[1]))
        valid = torch.arange(300, device=lens.device)[None, :] < lens[:, None]
        y = y.masked_fill(~valid[..., None], 0)
        et = text_encoder(captions).loc.float()
        er = motion_encoder(x.float(), lens).loc.float()
        eg = motion_encoder(y.float(), lens).loc.float()
        for em, hits, is_real in [(eg, correct, False), (er, real_correct, True)]:
            distances = torch.cdist(et, em)
            rank = distances.argsort(dim=1)
            truth = torch.arange(len(lens), device=lens.device)[:, None]
            hit = (rank[:, :3] == truth).cumsum(1).clamp_max(1).sum(0).cpu().numpy()
            hits += hit
            score = distances.diagonal().sum().item()
            if is_real:
                real_matching += score
            else:
                matching += score
        count += len(lens)
        real.append(er.cpu().numpy())
        generated.append(eg.cpu().numpy())
    real, generated = np.concatenate(real), np.concatenate(generated)
    mu_r, cov_r = calculate_activation_statistics(real)
    mu_g, cov_g = calculate_activation_statistics(generated)
    diversity_pairs = min(300, count - 1)
    result = dict(split=split, repeat=repeat, samples=count, excluded_tail=len(dataset)-count,
                  retrieval_pool=32, fid=float(calculate_frechet_distance(mu_r, cov_r, mu_g, cov_g)),
                  top1=float(correct[0]/count), top2=float(correct[1]/count), top3=float(correct[2]/count),
                  matching_distance=matching/count, diversity=float(calculate_diversity(generated, diversity_pairs)),
                  real_diversity=float(calculate_diversity(real, diversity_pairs)),
                  real_top1=float(real_correct[0]/count), real_top2=float(real_correct[1]/count),
                  real_top3=float(real_correct[2]/count), real_matching_distance=real_matching/count,
                  embedding_normalized=False, dataset_report=dataset.report,
                  num_res_layers=num_res_layers if residual is not None else (0 if mask is not None else 'all'))
    print('EVAL_METRICS '+json.dumps(result, ensure_ascii=False), flush=True)
    del text_encoder, motion_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize(rows):
    keys = ('fid', 'top1', 'top2', 'top3', 'matching_distance', 'diversity', 'real_diversity')
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    ci95=float(1.96*np.std([r[k] for r in rows], ddof=0)/np.sqrt(len(rows)))) for k in keys}
