"""BABEL + native MoMask RVQ on 272-d HumanML3D. Single GPU experiment runner."""
import argparse
import contextlib
import csv
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import bootstrap  # noqa: F401
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from components import SemanticRVQ, build_vq, build_transformer, pack_ema, restore_ema
from data272 import Motion272


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stage', choices=['check', 'vq', 'mask', 'res', 'eval'], required=True)
    p.add_argument('--data-root', default='/mnt/user/lh3/MotionStreamer-main/humanml3d_272')
    p.add_argument('--babel-dir', default=None)
    p.add_argument('--evaluator', default='/mnt/user/lh3/MotionStreamer-main/Evaluator_272/epoch=99.ckpt')
    p.add_argument('--eval-text-model', default='distilbert-base-uncased')
    p.add_argument('--run-dir', default='runs/lsa_momask272_s3407')
    p.add_argument('--lsa-weight', type=float, default=0.001)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch-size', default='auto', help='auto or positive integer; use same value in matched ablations')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--milestones', type=int, nargs='+', default=[50000, 80000])
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    p.add_argument('--memory-fraction', type=float, default=0.88, help='Auto-probe safety ceiling of total GPU memory')
    p.add_argument('--max-auto-batch', type=int, default=1024)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=10000, help='Validation TMR272 interval; 0 disables periodic evaluation')
    p.add_argument('--val-batches', type=int, default=32)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--sample-steps', type=int, default=18)
    p.add_argument('--cfg', type=float, default=4.)
    p.add_argument('--res-cfg', type=float, default=5.)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--repeat', type=int, default=20)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--vq-checkpoint', default=None)
    p.add_argument('--mask-checkpoint', default=None)
    p.add_argument('--res-checkpoint', default=None)
    p.add_argument('--num-res-layers', type=int, default=-1, help='eval only: -1=all; 0=base code only')
    p.add_argument('--init-vq', default=None, help='Explicit matching 272-d RVQ checkpoint, not a continuous SAE')
    p.add_argument('--code-dim', type=int, default=512)
    p.add_argument('--nb-code', type=int, default=512)
    p.add_argument('--num-quantizers', type=int, default=6)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--mu', type=float, default=0.99)
    p.add_argument('--quantize-dropout-prob', type=float, default=0.2)
    p.add_argument('--commit-weight', type=float, default=0.02)
    p.add_argument('--position-weight', type=float, default=0.5)
    p.add_argument('--latent-dim', type=int, default=384)
    p.add_argument('--ff-size', type=int, default=1024)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--dropout', type=float, default=0.2)
    a = p.parse_args()
    if a.steps < 1 or a.workers < 0 or a.repeat < 1 or a.save_every < 1 or a.val_batches < 1:
        p.error('steps/repeat/save-every/val-batches must be positive; workers nonnegative')
    if a.num_quantizers < 2 or a.lsa_weight < 0 or not 0 < a.memory_fraction < 1:
        p.error('num-quantizers >=2, lsa-weight >=0 and memory-fraction in (0,1) required')
    if a.num_res_layers < -1 or a.num_res_layers >= a.num_quantizers:
        p.error('num-res-layers must be -1 or 0..num-quantizers-1')
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def architecture(a):
    return {k: getattr(a, k) for k in ('code_dim', 'nb_code', 'num_quantizers', 'width', 'depth', 'mu',
            'quantize_dropout_prob', 'latent_dim', 'ff_size', 'layers', 'heads', 'dropout')}


def state_without_clip(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('clip_model.')}


def load_transformer(a, stage, checkpoint):
    ck = load_checkpoint(checkpoint)
    model = build_transformer(ck['config'], stage, a.device)
    missing, extra = model.load_state_dict(ck['model'], strict=False)
    if extra or any(not k.startswith('clip_model.') for k in missing):
        raise RuntimeError(f'Invalid {stage} checkpoint: missing={missing}, extra={extra}')
    return model, ck


def resolve_vq(a):
    # Select by validation FID when available, otherwise by validation reconstruction loss.
    base = Path(a.run_dir) / 'vq'
    path = Path(a.vq_checkpoint) if a.vq_checkpoint else (base / 'best_fid.pt' if (base / 'best_fid.pt').exists() else base / 'best.pt')
    ck = load_checkpoint(path)
    if float(ck['args']['lsa_weight']) != a.lsa_weight:
        raise ValueError('LSA weight differs from the selected tokenizer; set --lsa-weight to its training value')
    model = build_vq(ck['config']).to(a.device)
    model.load_state_dict(ck['vq_model'], strict=True)
    model.eval().requires_grad_(False)
    return model, ck['config'], str(path.resolve())


def make_loader(a, split, training, window=0, semantic=False):
    ds = Motion272(a.data_root, split, training=training, window=window,
                   semantic=semantic, babel_dir=a.babel_dir)
    kwargs = dict(batch_size=int(a.batch_size), shuffle=training, num_workers=a.workers,
                  pin_memory=a.device.startswith('cuda'), drop_last=False,
                  worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(a.seed))
    if a.workers:
        kwargs.update(persistent_workers=True, prefetch_factor=a.prefetch_factor)
    return DataLoader(ds, **kwargs)


def amp_context(a):
    if a.precision == 'fp32':
        return contextlib.nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16 if a.precision == 'bf16' else torch.float16)


def loss_forward(a, model, vq, batch):
    x = batch['motion'].to(a.device, non_blocking=True)
    lens = batch['length'].to(a.device)
    if a.stage == 'vq':
        babel = batch['babel'].to(a.device, non_blocking=True) if a.lsa_weight else None
        has = batch['has_babel'].to(a.device)
        with amp_context(a):
            y, commit, ppl, sem = model(x, babel, lens, has)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None]
        rec = F.smooth_l1_loss(y.float(), x, reduction='none').mean(-1)
        rec = (rec * valid).sum() / valid.sum()
        # 272-d layout: 8 root channels followed by 22*3 local positions.
        pos = F.smooth_l1_loss(y[..., 8:74].float(), x[..., 8:74], reduction='none').mean(-1)
        pos = (pos * valid).sum() / valid.sum()
        loss = rec + a.position_weight*pos + a.commit_weight*commit + a.lsa_weight*sem
        return loss, dict(recon=float(rec.detach()), position=float(pos.detach()),
                          semantic=float(sem.detach()), perplexity=float(ppl.detach()))
    with torch.no_grad():
        ids, _ = vq.encode(x)
    with amp_context(a):
        loss, _, acc = model(ids[..., 0] if a.stage == 'mask' else ids,
                             list(batch['caption']), lens // 4)
    return loss, dict(accuracy=float(acc))


def tune_batch(a):
    # Each trial is an isolated process: no probe EMA/optimizer state leaks into training.
    candidates = [b for b in (16, 32, 64, 128, 256, 384, 512, 768, 1024) if b <= a.max_auto_batch]
    command = [sys.executable, str(Path(__file__).resolve())]
    raw = sys.argv[1:]
    keep, skip = [], False
    for item in raw:
        if skip:
            skip = False
            continue
        if item == '--batch-size':
            skip = True
            continue
        if item.startswith('--batch-size='):
            continue
        keep.append(item)
    results = []
    for batch in candidates:
        print(f'GPU batch probe: {a.stage} batch={batch}', flush=True)
        run = subprocess.run(command+keep+['--batch-size', str(batch), '--probe'],
                             capture_output=True, text=True)
        markers = [line for line in run.stdout.splitlines() if line.startswith('PROBE_RESULT=')]
        if run.returncode:
            message = run.stderr + run.stdout
            if 'out of memory' in message.lower():
                print(f'Probe batch={batch}: OOM; stopping probe.', flush=True)
                break
            raise RuntimeError(f'Batch probe failed (not OOM):\n{message[-6000:]}')
        if not markers:
            raise RuntimeError('Probe did not return memory/throughput measurements')
        result = json.loads(markers[-1].split('=', 1)[1])
        print(json.dumps(result), flush=True)
        results.append(result)
        if result['reserved_fraction'] > a.memory_fraction:
            break
    safe = [r for r in results if r['reserved_fraction'] <= a.memory_fraction]
    if not safe:
        raise RuntimeError('No batch fits safety ceiling; free GPU memory or lower --max-auto-batch / explicit batch size')
    # High occupancy is a means, not a goal: choose measured throughput under ceiling.
    chosen = max(safe, key=lambda r: r['samples_per_second'])
    a.batch_size = str(chosen['batch_size'])
    out = Path(a.run_dir) / a.stage
    out.mkdir(parents=True, exist_ok=True)
    (out / 'batch_probe.json').write_text(json.dumps(dict(trials=results, selected=chosen), indent=2))
    print(f'Selected batch {a.batch_size}; {chosen["samples_per_second"]:.1f} samples/s', flush=True)


def validate(a, model, vq):
    loader = make_loader(a, 'val', False, window=64 if a.stage == 'vq' else 0,
                         semantic=a.stage == 'vq' and a.lsa_weight > 0)
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Validation loss', total=min(len(loader), a.val_batches), dynamic_ncols=True)):
            loss, details = loss_forward(a, model, vq, batch)
            value = details['recon'] + a.position_weight*details['position'] if a.stage == 'vq' else float(loss)
            total += value * len(batch['motion'])
            count += len(batch['motion'])
            if i+1 >= a.val_batches:
                break
    return total / count


def train(a):
    out = Path(a.run_dir) / a.stage
    latest = out / 'latest.pt'
    if not a.probe and latest.exists() and not a.resume:
        raise FileExistsError(f'{latest} exists; use --resume or a new --run-dir')
    ck = load_checkpoint(latest) if a.resume and latest.exists() else None
    if a.resume and ck is None:
        raise FileNotFoundError(f'Cannot resume: {latest}')
    if a.stage == 'vq':
        config = ck['config'] if ck else architecture(a)
        model = SemanticRVQ(config).to(a.device)
        vq, vq_path = None, None
        if a.init_vq and ck is None:
            initial = load_checkpoint(a.init_vq)
            weights = initial.get('vq_model', initial.get('net'))
            if weights is None:
                raise ValueError('--init-vq expects vq_model or net weights of a matching 272-d RVQ')
            model.vq.load_state_dict(weights, strict=True)
            # Explicit warm start: initialize EMA statistics from loaded codebooks.
            for layer in model.vq.quantizer.layers:
                layer.init = True
                layer.code_sum = layer.codebook.clone()
                layer.code_count = torch.ones(layer.nb_code, device=a.device)
    else:
        vq, vq_config, vq_path = resolve_vq(a)
        config = ck['config'] if ck else {**architecture(a), **{k:vq_config[k] for k in ('code_dim','nb_code','num_quantizers','width','depth','mu','quantize_dropout_prob')}}
        model = build_transformer(config, a.stage, a.device)
    if vq_path:
        from evaluation import file_hash
        vq_digest = file_hash(vq_path)
    else:
        vq_digest = None
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=a.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=a.precision == 'fp16')
    start, best, best_fid = 0, math.inf, math.inf
    if ck:
        # Optimizer/scheduler semantics must not silently change on resume.
        for key in ('lsa_weight','seed','lr','warmup','milestones','gamma','weight_decay','precision','data_root','position_weight','commit_weight','batch_size'):
            if str(ck['args'][key]) != str(getattr(a, key)):
                raise ValueError(f'Resume configuration mismatch for {key}: {ck["args"][key]} vs {getattr(a,key)}')
        if a.stage != 'vq' and ck['vq_checkpoint'] != vq_path:
            raise ValueError('Resume would change the frozen tokenizer checkpoint')
        if a.stage != 'vq' and ck['vq_sha256'] != vq_digest:
            raise ValueError('Frozen tokenizer contents changed since the saved training checkpoint')
        if a.stage == 'vq':
            model.load_state_dict(ck['model'], strict=True)
            restore_ema(model.vq, ck['ema'], a.device)
        else:
            missing, extra = model.load_state_dict(ck['model'], strict=False)
            if extra or any(not k.startswith('clip_model.') for k in missing):
                raise RuntimeError('Invalid transformer state')
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        start, best, best_fid = ck['step'], ck['best'], ck.get('best_fid', math.inf)
    loader = make_loader(a, 'train', True, window=64 if a.stage == 'vq' else 0,
                         semantic=a.stage == 'vq' and a.lsa_weight > 0)
    iterator = iter(loader)
    if not a.probe:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(dict(args=vars(a), config=config, vq_checkpoint=vq_path,
                                                       dataset=loader.dataset.report), indent=2))
    if ck:
        random.setstate(ck['rng']['python'])
        np.random.set_state(ck['rng']['numpy'])
        torch.set_rng_state(ck['rng']['torch'])
        if ck['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    if a.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    num_steps = 4 if a.probe else a.steps
    began = time.monotonic()
    measured_start = began
    model.train()
    if a.stage != 'vq':
        model.clip_model.eval()
    bar = tqdm(range(start, num_steps), desc=f'{a.stage} training', initial=start, total=num_steps,
               dynamic_ncols=True, disable=a.probe)
    log_path = out / 'train.csv'

    def save(step, name):
        value = dict(model=state_without_clip(model), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     config=config, args=vars(a), step=step, best=best, best_fid=best_fid, vq_checkpoint=vq_path,
                     vq_sha256=vq_digest,
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        if a.stage == 'vq':
            value.update(vq_model=model.vq.state_dict(), ema=pack_ema(model.vq))
        atomic_save(value, out / name)

    for step0 in bar:
        step = step0 + 1
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        lr = a.lr * min(1., step / max(1, a.warmup)) * a.gamma**sum(step >= m for m in a.milestones)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, details = loss_forward(a, model, vq, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at {a.stage} step {step}: {details}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if a.probe:
            if a.device.startswith('cuda'):
                torch.cuda.synchronize()
            if step == 1:
                measured_start = time.monotonic()
            continue
        elapsed = time.monotonic() - began
        eta = elapsed / max(1, step-start) * max(0, a.steps-step)
        gpu_gb = torch.cuda.max_memory_allocated()/2**30 if a.device.startswith('cuda') else 0
        loss_value = float(loss.detach())
        bar.set_postfix(loss=f'{loss_value:.4f}', lr=f'{lr:.2g}', GPU=f'{gpu_gb:.1f}GiB', ETA=f'{eta/3600:.2f}h')
        if step % 20 == 0 or step == 1:
            row = dict(step=step, loss=loss_value, lr=lr, elapsed_s=elapsed, eta_s=eta, gpu_peak_gib=gpu_gb, **details)
            exists = log_path.exists()
            with log_path.open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        if step % a.save_every == 0 or step == a.steps:
            optimizer.zero_grad(set_to_none=True)
            value = validate(a, model, vq)
            if value < best:
                best = value
                save(step, 'best.pt')
            save(step, 'latest.pt')
            model.train()
            if a.stage != 'vq':
                model.clip_model.eval()
        if (a.eval_every and step % a.eval_every == 0) or step == a.steps:
            from evaluation import evaluate
            optimizer.zero_grad(set_to_none=True)
            mask, residual = None, None
            if a.stage == 'mask':
                mask = model
            elif a.stage == 'res':
                mask_path = a.mask_checkpoint or str(Path(a.run_dir) / 'mask/best_fid.pt')
                if not Path(mask_path).is_file():
                    mask_path = str(Path(a.run_dir) / 'mask/best.pt')
                mask, mask_ck = load_transformer(a, 'mask', mask_path)
                if mask_ck['vq_checkpoint'] != vq_path:
                    raise ValueError('Mask and residual models use different tokenizer checkpoints')
                if mask_ck['vq_sha256'] != vq_digest:
                    raise ValueError('Mask model was trained with different tokenizer contents')
                residual = model
            result = evaluate(a, model.vq if a.stage == 'vq' else vq, mask, residual, split='val')
            result['step'] = step
            with (out / 'val_metrics.jsonl').open('a') as f:
                f.write(json.dumps(result)+'\n')
            if result['fid'] < best_fid:
                best_fid = result['fid']
                save(step, 'best_fid.pt')
            save(step, 'latest.pt')
            if a.stage == 'res':
                del mask
            gc.collect()
            if a.device.startswith('cuda'):
                torch.cuda.empty_cache()
            model.train()
            if a.stage != 'vq':
                model.clip_model.eval()
    if a.probe:
        total = torch.cuda.get_device_properties(a.device).total_memory
        reserved = torch.cuda.max_memory_reserved()
        free_now, _ = torch.cuda.mem_get_info()
        # Account for memory belonging to other processes as well.
        outside = max(0, total-free_now-torch.cuda.memory_reserved())
        print('PROBE_RESULT='+json.dumps(dict(batch_size=int(a.batch_size), reserved_gib=reserved/2**30,
              reserved_fraction=(reserved+outside)/total, samples_per_second=3*len(batch['motion'])/(time.monotonic()-measured_start))))


def check(a):
    import importlib.util
    for dependency in ('einops', 'scipy', 'tqdm', 'transformers', 'pytorch_lightning', 'clip'):
        if importlib.util.find_spec(dependency) is None:
            raise ImportError(f'Missing {dependency}; see requirements-extra.txt / README')
    import clip
    if not hasattr(clip, 'load') or not hasattr(clip, 'tokenize'):
        raise ImportError('Wrong clip package: install the OpenAI CLIP repository, not PyPI clip')
    if not Path(a.evaluator).is_file():
        raise FileNotFoundError(f'Evaluator checkpoint not found: {a.evaluator}; override --evaluator')
    for split in ('train', 'val', 'test'):
        dataset = Motion272(a.data_root, split, semantic=split == 'train' and a.lsa_weight > 0, babel_dir=a.babel_dir)
    # Fail before a long training run if evaluator weights/backbone are incompatible.
    from evaluation import load_evaluator
    text_encoder, motion_encoder = load_evaluator(a.evaluator, a.eval_text_model, a.device)
    sample = dataset[0]
    with torch.no_grad():
        text = text_encoder([sample['caption']]).loc
        motion = motion_encoder(sample['motion'][None].to(a.device), torch.tensor([sample['length']], device=a.device)).loc
    if text.shape != motion.shape or not torch.isfinite(text).all() or not torch.isfinite(motion).all():
        raise ValueError('Evaluator returned invalid or incompatible text/motion embeddings')
    print('Paths, dimensions, BABEL alignment, evaluator weights and embedding forward pass checked.')


def main():
    a = parse_args()
    seed_everything(a.seed)
    if a.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Python environment')
        torch.cuda.set_device(a.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        if a.precision == 'auto':
            a.precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    else:
        a.precision = 'fp32'
    if a.stage == 'check':
        check(a)
        return
    if not Path(a.evaluator).is_file() and not a.probe:
        raise FileNotFoundError(f'Evaluator checkpoint not found: {a.evaluator}')
    if a.stage == 'eval':
        from evaluation import evaluate, summarize, file_hash
        vq, config, vq_path = resolve_vq(a)
        vq_digest = file_hash(vq_path)
        def selected(stage, override):
            base = Path(a.run_dir) / stage
            return override or str(base / ('best_fid.pt' if (base/'best_fid.pt').is_file() else 'best.pt'))
        mask_path = selected('mask', a.mask_checkpoint)
        mask, mask_ck = load_transformer(a, 'mask', mask_path)
        if a.num_res_layers == 0:
            residual, res_path = None, None
        else:
            res_path = selected('res', a.res_checkpoint)
            residual, res_ck = load_transformer(a, 'res', res_path)
            if res_ck['vq_checkpoint'] != vq_path:
                raise ValueError('Residual checkpoint tokenizer mismatch')
            if res_ck['vq_sha256'] != vq_digest:
                raise ValueError('Residual checkpoint tokenizer contents mismatch')
        if mask_ck['vq_checkpoint'] != vq_path:
            raise ValueError('Mask checkpoint tokenizer mismatch')
        if mask_ck['vq_sha256'] != vq_digest:
            raise ValueError('Mask checkpoint tokenizer contents mismatch')
        rows = []
        for repeat in range(a.repeat):
            seed_everything(a.seed + repeat)
            rows.append(evaluate(a, vq, mask, residual, split=a.split, repeat=repeat, num_res_layers=a.num_res_layers))
        output = dict(args=vars(a), rows=rows, summary=summarize(rows),
                      uncertainty='95% normal CI over generation/evaluation repeats, NOT independent training runs',
                      checkpoints=dict(vq=vq_path, mask=mask_path, residual=res_path),
                      evaluator_sha256=file_hash(a.evaluator),
                      mean_sha256=file_hash(Path(a.data_root)/'mean_std/Mean.npy'),
                      std_sha256=file_hash(Path(a.data_root)/'mean_std/Std.npy'))
        dest = Path(a.run_dir) / f'eval_{a.split}_res{a.num_res_layers}.json'
        dest.write_text(json.dumps(output, indent=2))
        csv_path = dest.with_suffix('.csv')
        flat = dict(split=a.split, seed=a.seed, lsa_weight=a.lsa_weight, num_res_layers=a.num_res_layers,
                    repeats=a.repeat, **{f'{key}_{stat}':value for key, stats in output['summary'].items() for stat,value in stats.items()})
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)
        print(json.dumps(output['summary'], indent=2))
        print(f'Saved {dest}')
        return
    if a.batch_size == 'auto':
        if a.resume:
            a.batch_size = str(load_checkpoint(Path(a.run_dir)/a.stage/'latest.pt')['args']['batch_size'])
        elif a.device.startswith('cuda'):
            tune_batch(a)
        else:
            a.batch_size = '2'
    if int(a.batch_size) < 1:
        raise ValueError('batch-size must be positive')
    train(a)


if __name__ == '__main__':
    main()
