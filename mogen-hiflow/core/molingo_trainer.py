import sys
sys.path.append('.')

import copy
import csv
import math
import os
from os.path import join as pjoin

import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

import mogen.utils.lr_sched as lr_sched
import mogen.utils.misc as misc
from mogen.core.eval import (
    calculate_R_precision,
    euclidean_distance_matrix,
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from mogen.utils.misc import NativeScalerWithGradNormCount as NativeScaler


def is_main():
    return (not dist.is_available() or not dist.is_initialized()
            or dist.get_rank() == 0)


def def_value():
    return 0.0


def lengths_to_mask(lengths, device, max_len) -> torch.Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


@torch.no_grad()
def _evaluate_molingo_once(val_loader, model_without_ddp, vae_model, ema_params, ep, cfg,
                           temperature, motionencoder, textencoder, rank=0,
                           use_ema=True, std_factor=1., acc_ratio=1.):
    model_without_ddp.eval()
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    tmr_real_list, tmr_pred_list = [], []
    RP_tmr_real, RP_tmr_pred = 0, 0
    matching_score_tmr_real, matching_score_tmr_pred = 0, 0
    nb_sample = 0

    for _i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Eval {ep}", leave=False):
        clip_text, feats_ref, m_length = batch
        feats_ref = feats_ref.detach().cuda().float()
        m_length = m_length.cuda()
        bs, _token_len = feats_ref.shape[:2]

        sampled_tokens = model_without_ddp.sample_tokens(bsz=bs, m_lens=m_length, cfg=cfg,
                                                         cfg_schedule="linear", labels=clip_text,
                                                         temperature=temperature, acc_ratio=acc_ratio)
        feats_rst = vae_model.decode(sampled_tokens / std_factor)

        et_tmr_pred, em_tmr_pred = textencoder(clip_text).loc, motionencoder(feats_rst, m_length).loc
        et_tmr, em_tmr = textencoder(clip_text).loc, motionencoder(feats_ref, m_length).loc

        tmr_real_list.append(em_tmr)
        tmr_pred_list.append(em_tmr_pred)

        tmr_R_real = calculate_R_precision(et_tmr.cpu().numpy(), em_tmr.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_real = euclidean_distance_matrix(et_tmr.cpu().numpy(), em_tmr.cpu().numpy()).trace()
        RP_tmr_real += tmr_R_real
        matching_score_tmr_real += tmr_match_real

        tmr_R_pred = calculate_R_precision(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy(), top_k=3, sum_all=True)
        tmr_match_pred = euclidean_distance_matrix(et_tmr_pred.cpu().numpy(), em_tmr_pred.cpu().numpy()).trace()
        RP_tmr_pred += tmr_R_pred
        matching_score_tmr_pred += tmr_match_pred

        nb_sample += bs

    tmr_real_list_np = torch.cat(tmr_real_list, dim=0).cpu().numpy()
    tmr_pred_list_np = torch.cat(tmr_pred_list, dim=0).cpu().numpy()

    mu_tmr_real, cov_tmr_real = calculate_activation_statistics(tmr_real_list_np)
    mu_tmr_pred, cov_tmr_pred = calculate_activation_statistics(tmr_pred_list_np)

    fid_tmr = calculate_frechet_distance(mu_tmr_real, cov_tmr_real, mu_tmr_pred, cov_tmr_pred)

    RP_tmr_real = RP_tmr_real / nb_sample
    RP_tmr_pred = RP_tmr_pred / nb_sample
    matching_score_tmr_real = matching_score_tmr_real / nb_sample
    matching_score_tmr_pred = matching_score_tmr_pred / nb_sample

    if rank == 0:
        wandb.log({
            "tmr/fid": fid_tmr,
            "tmr/r1": RP_tmr_pred[0],
            "tmr/r2": RP_tmr_pred[1],
            "tmr/r3": RP_tmr_pred[2],
            "tmr/mscore": matching_score_tmr_pred,
        }, step=ep)

    msg_tmr = (f"--> \t Ep {ep} :, FID_TMR. {fid_tmr:.4f}, RP_TMR_real. {RP_tmr_real}, "
               f"RP_tmr_pred. {RP_tmr_pred}, matching_score_tmr_real. {matching_score_tmr_real}, "
               f"matching_score_tmr_pred. {matching_score_tmr_pred}")
    print(msg_tmr)

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    return fid_tmr, RP_tmr_pred[0], RP_tmr_pred[1], RP_tmr_pred[2], matching_score_tmr_pred


def _append_eval_metrics(model_dir, row):
    metrics_path = pjoin(model_dir, "eval_metrics.txt")
    fieldnames = [
        "epoch", "fid", "top1", "top2", "top3", "matching_score",
        "is_best_fid", "is_best_top1", "is_best_top2", "is_best_top3", "is_best_matching",
        "best_fid_before", "best_top1_before", "best_top2_before", "best_top3_before",
        "best_matching_before", "lr",
    ]
    write_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class MoLingoTrainer:
    def __init__(self, args, model, model_without_ddp, vae_model, datamodule, rank=0, ae=False):
        self.opt = args
        self.model = model
        self.ae = ae
        self.model_without_ddp = model_without_ddp
        self.vae_model = vae_model
        self.device = args.device
        self.datamodule = datamodule
        self.distributed = args.distributed
        self.vae_model.eval()
        self.rank = rank

        if rank == 0:
            wandb.init(
                project='molingo-ms',
                name=self.opt.name,
            )

    def forward(self, batch_data, plot_t2m, std_factor):
        conds, motion, m_lens = batch_data

        motion = motion.detach().float().to(self.device)

        with torch.no_grad():
            if not self.ae:
                x, _dist = self.vae_model.encode(motion)
            else:
                x = self.vae_model.ae_encode(motion)
            x = x.mul_(std_factor)

        with torch.cuda.amp.autocast():
            loss, loss_dict = self.model(x, conds, m_lens)

        return loss, loss_dict

    def train(self, train_loader, eval_val_loader, motionencoder, textencoder, plot_eval):
        self.model.to(self.device)
        self.vae_model.to(self.device)

        model_without_ddp = self.model_without_ddp
        loss_scaler = NativeScaler()
        base_lr = self.opt.base_lr

        param_groups = misc.add_weight_decay(model_without_ddp, 0.02)
        self.optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95))

        epoch = 0
        it = 0

        resume_path = pjoin(self.opt.model_dir, "net_best_fid.pth")
        if os.path.exists(resume_path):
            torch.cuda.empty_cache()
            checkpoint = torch.load(resume_path, map_location='cpu')
            print(f"Loading from resume path {resume_path}")
            model_without_ddp.load_state_dict(checkpoint['model'])
            model_params = list(model_without_ddp.parameters())
            ema_state_dict = checkpoint['model_ema']
            ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            print("Resume checkpoint %s" % resume_path)
            torch.cuda.empty_cache()
            del checkpoint
        else:
            model_params = list(model_without_ddp.parameters())
            ema_params = copy.deepcopy(model_params)
            print("Training from scratch")

        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(eval_val_loader)))

        best_fid_tmr = 5000.
        best_top1_tmr, best_top2_tmr, best_top3_tmr, best_matching_score_tmr = 0., 0., 0., 100.
        progress_bar = tqdm(total=total_iters, initial=it, desc="Training", disable=not is_main())

        while epoch < self.opt.max_epoch:
            if self.opt.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.model.train()
            torch.backends.cudnn.deterministic = False
            self.vae_model.eval()

            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

            self.optimizer.zero_grad()

            loss_value = 0.0
            for i, batch in enumerate(train_loader):
                it += 1
                lr_sched.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, base_lr, self.opt)

                loss, loss_dict = self.forward(batch, plot_eval, std_factor=self.opt.std_factor)
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                loss_scaler(loss, self.optimizer, clip_grad=3.0, parameters=self.model.parameters(), update_grad=True)
                self.optimizer.zero_grad()

                torch.cuda.synchronize()

                update_ema(ema_params, model_params, rate=0.9999)

                metric_logger.update(loss=loss_value)
                lr = self.optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

                if is_main():
                    progress_bar.update(1)
                    if it % 100 == 0:
                        progress_bar.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.6g}")

            if is_main():
                wandb.log({"train/loss": loss_value}, step=epoch)

            metric_logger.synchronize_between_processes()
            print(f'epoch {epoch}  Averaged stats: {metric_logger}')

            epoch += 1

            eval_start_epoch = getattr(self.opt, "eval_start_epoch", 100)
            if epoch >= eval_start_epoch and epoch % self.opt.eval_every_e == 0:
                best_fid_before = best_fid_tmr
                best_top1_before = best_top1_tmr
                best_top2_before = best_top2_tmr
                best_top3_before = best_top3_tmr
                best_matching_before = best_matching_score_tmr

                fid_tmr, top1_tmr, top2_tmr, top3_tmr, matching_score_tmr = _evaluate_molingo_once(
                    eval_val_loader, model_without_ddp, self.vae_model, ema_params, epoch,
                    cfg=self.opt.cfg, temperature=self.opt.temperature,
                    motionencoder=motionencoder, textencoder=textencoder,
                    rank=self.rank, std_factor=self.opt.std_factor,
                    acc_ratio=self.opt.acc_ratio,
                )

                is_best_fid = fid_tmr < best_fid_tmr
                is_best_top1 = top1_tmr > best_top1_tmr
                is_best_top2 = top2_tmr > best_top2_tmr
                is_best_top3 = top3_tmr > best_top3_tmr
                is_best_matching = matching_score_tmr < best_matching_score_tmr
                out_model_dir = pjoin(self.opt.save_root, 'model')

                if is_best_fid:
                    print(f"--> --> \t FID_TMR Improved from {best_fid_tmr:.5f} to {fid_tmr:.5f} !!!")
                    best_fid_tmr = fid_tmr
                    misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_fid_tmr")
                    misc.save_model_simplified(self.opt.model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_fid")

                if is_best_matching:
                    print(f"--> --> \t matching_score TMR Improved from {best_matching_score_tmr:.5f} "
                          f"to {matching_score_tmr:.5f} !!!")
                    best_matching_score_tmr = matching_score_tmr
                    misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_matching_tmr")

                if is_best_top1:
                    print(f"--> --> \t Top1 TMR Improved from {best_top1_tmr:.4f} to {top1_tmr:.4f} !!!")
                    best_top1_tmr = top1_tmr
                    misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_r1_tmr")

                if is_best_top2:
                    print(f"--> --> \t Top2 TMR Improved from {best_top2_tmr:.4f} to {top2_tmr:.4f} !!!")
                    best_top2_tmr = top2_tmr
                    misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_r2_tmr")

                if is_best_top3:
                    print(f"--> --> \t Top3 TMR Improved from {best_top3_tmr:.4f} to {top3_tmr:.4f} !!!")
                    best_top3_tmr = top3_tmr
                    misc.save_model_simplified(out_model_dir, model_without_ddp=model_without_ddp, epoch=epoch,
                                               ema_params=ema_params, epoch_name="net_best_r3_tmr")

                if is_main():
                    _append_eval_metrics(self.opt.model_dir, {
                        "epoch": epoch,
                        "fid": fid_tmr,
                        "top1": top1_tmr,
                        "top2": top2_tmr,
                        "top3": top3_tmr,
                        "matching_score": matching_score_tmr,
                        "is_best_fid": int(is_best_fid),
                        "is_best_top1": int(is_best_top1),
                        "is_best_top2": int(is_best_top2),
                        "is_best_top3": int(is_best_top3),
                        "is_best_matching": int(is_best_matching),
                        "best_fid_before": best_fid_before,
                        "best_top1_before": best_top1_before,
                        "best_top2_before": best_top2_before,
                        "best_top3_before": best_top3_before,
                        "best_matching_before": best_matching_before,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })

            # Periodic latest checkpoints are intentionally disabled; only metric-best models are saved.

        if is_main():
            progress_bar.close()


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to source parameters via EMA."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
