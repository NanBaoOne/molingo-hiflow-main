from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from .embedding import lengths_to_mask
from .interp import interpolate_latents
from .scheduler_flow_matching import PyramidFlowMatchingScheduler
from .transformer import MotionFlux


class TextAdapter(nn.Module):
    def __init__(self, n_layers=6, d_model=1024, n_head=16, ff_mult=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_states, key_padding_mask=None):
        return self.norm(self.blocks(hidden_states, src_key_padding_mask=key_padding_mask))


class T5TextConditioner(nn.Module):
    def __init__(self, hidden_size, num_heads, adapter_layers=6, t5_max_len=64,
                 label_drop_prob=0.1, dropout=0.1):
        super().__init__()
        self.t5_max_len = t5_max_len
        self.label_drop_prob = label_drop_prob
        self.dummy_text = ""
        self.t5_tok = T5Tokenizer.from_pretrained("t5-large")
        self.t5_model = T5EncoderModel.from_pretrained("t5-large")
        self.t5_model.eval()
        for p in self.t5_model.parameters():
            p.requires_grad_(False)
        self.cond_proj = nn.Linear(1024, hidden_size)
        self.text_aligner = TextAdapter(adapter_layers, hidden_size, num_heads, dropout=dropout)
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

    def train(self, mode=True):
        super().train(mode)
        self.t5_model.eval()
        return self

    def mask_text(self, text_list, force_mask=False):
        bsz = len(text_list)
        if force_mask:
            return [self.dummy_text for _ in range(bsz)]
        if self.training and self.label_drop_prob > 0:
            text_list = list(text_list)
            random_mask = torch.rand(bsz) < self.label_drop_prob
            for i, masked in enumerate(random_mask):
                if masked:
                    text_list[i] = self.dummy_text
            return text_list
        return text_list

    def forward(self, raw_text, force_mask=False):
        raw_text = self.mask_text(raw_text, force_mask=force_mask)
        device = next(self.parameters()).device
        batch = self.t5_tok(
            text=raw_text,
            max_length=self.t5_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            hidden = self.t5_model(**batch).last_hidden_state
        text_padding_mask = ~batch["attention_mask"].bool()
        word_emb = self.cond_proj(hidden)
        word_emb = self.text_aligner(word_emb, text_padding_mask)
        valid = (~text_padding_mask).unsqueeze(-1).to(word_emb.dtype)
        pooled = (word_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.pool_proj(pooled)
        return word_emb, text_padding_mask, pooled


def parse_scales(scales):
    if isinstance(scales, str):
        scales = scales.strip().strip("()[]")
        scales = [scale for scale in scales.replace(";", ",").split(",") if scale.strip()]
    return tuple(float(scale) for scale in scales)


class ConvPatch1D(nn.Module):
    def __init__(self, vae_dim, latent_dim, time_patch):
        super().__init__()
        self.time_patch = time_patch
        self.net = nn.Sequential(
            nn.ConstantPad1d((0, time_patch - 1), 0.0),
            nn.Conv1d(vae_dim, latent_dim, kernel_size=time_patch, stride=time_patch),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class HiFlowSAEModel(nn.Module):
    def __init__(self, vae_dim, latent_dim, depth, num_heads, time_patch, mlp_ratio=4.0,
                 adapter_layers=6, t5_max_len=64, label_drop_prob=0.1, proj_dropout=0.1):
        super().__init__()
        self.time_patch = time_patch
        self.vae_dim = vae_dim
        self.latent_dim = latent_dim
        self.input_process = ConvPatch1D(vae_dim, latent_dim, time_patch)
        self.text_conditioner = T5TextConditioner(
            latent_dim, num_heads, adapter_layers=adapter_layers, t5_max_len=t5_max_len,
            label_drop_prob=label_drop_prob, dropout=proj_dropout,
        )
        self.transformer = MotionFlux(
            input_size=latent_dim,
            output_size=vae_dim * time_patch,
            hidden_size=latent_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=proj_dropout,
        )

    def unpatch(self, x, target_len):
        bsz, patch_len, _ = x.shape
        x = x.view(bsz, patch_len, self.time_patch, self.vae_dim)
        x = x.reshape(bsz, patch_len * self.time_patch, self.vae_dim)
        return x[:, :target_len]

    def forward(self, x, timesteps, labels, m_lens, force_mask=False, scale_value=None):
        device = x.device
        m_lens = m_lens.to(device)
        timesteps = timesteps.to(device)
        target_len = x.shape[1]
        x_patch = self.input_process(x)
        patch_len = x_patch.shape[1]
        latent_lens = torch.div(m_lens + self.time_patch - 1, self.time_patch, rounding_mode="floor")
        motion_padding_mask = ~lengths_to_mask(latent_lens.clamp(max=patch_len), patch_len).to(device)
        word_emb, text_padding_mask, pooled = self.text_conditioner(labels, force_mask=force_mask)
        word_emb = word_emb.to(device)
        text_padding_mask = text_padding_mask.to(device)
        pooled = pooled.to(device)
        motion_ids = torch.arange(patch_len, device=device, dtype=torch.float32).view(1, patch_len, 1).expand(x.shape[0], -1, -1)
        scale_values = None
        if scale_value is not None:
            scale_values = torch.full((x.shape[0],), float(scale_value), device=device, dtype=torch.float32)
            motion_ids = motion_ids / scale_values.view(-1, 1, 1).clamp_min(1e-6)
        text_ids = torch.zeros(x.shape[0], word_emb.shape[1], 1, dtype=torch.float32, device=device)
        out = self.transformer(
            x_patch, timesteps, word_emb, pooled, motion_ids, text_ids,
            motion_mask=motion_padding_mask, text_mask=text_padding_mask,
            scale_values=scale_values,
        )
        return self.unpatch(out, target_len)


class HiFlowSAE(nn.Module):
    def __init__(self, decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4.0, vae_embed_dim=256, label_drop_prob=0.1, proj_dropout=0.1,
                 unit_length=4, grad_checkpointing=False, token_size=75, sample_steps=32,
                 t5_max_len=64, adapter_layers=6, ae=False, time_patch=2,
                 scales=(0.3, 0.6, 1.0), **kwargs):
        super().__init__()
        self.ae = ae
        self.seq_len = token_size
        self.token_embed_dim = vae_embed_dim
        self.unit_length = unit_length
        self.sample_steps = sample_steps
        self.time_patch = time_patch
        self.scales = parse_scales(scales)
        self.scheduler = PyramidFlowMatchingScheduler(sample_steps, scales=self.scales)
        self.model = HiFlowSAEModel(
            vae_dim=vae_embed_dim,
            latent_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            time_patch=time_patch,
            mlp_ratio=mlp_ratio,
            adapter_layers=adapter_layers,
            t5_max_len=t5_max_len,
            label_drop_prob=label_drop_prob,
            proj_dropout=proj_dropout,
        )

    def _stage_lens(self, latent_lens, stage_id):
        scale = self.scales[stage_id]
        return torch.ceil(latent_lens.float() * scale).long().clamp(min=1, max=self.seq_len).to(latent_lens.device)

    def _make_stage_latent(self, x, latent_lens, stage_lens):
        latent_lens = latent_lens.to(x.device)
        stage_lens = stage_lens.to(x.device)
        stage_max_len = int(stage_lens.max().item())
        stage = x.new_zeros(x.shape[0], stage_max_len, x.shape[2])
        for idx in range(x.shape[0]):
            src_len = int(latent_lens[idx].item())
            dst_len = int(stage_lens[idx].item())
            src = x[idx:idx + 1, :src_len]
            if src_len != dst_len:
                src = interpolate_latents(src, size=dst_len)
            stage[idx, :dst_len] = src[0]
        mask = lengths_to_mask(stage_lens, stage_max_len).to(x.device)
        return torch.where(mask.unsqueeze(-1), stage, torch.zeros_like(stage)), mask

    def _upsample_stage(self, x, src_lens, dst_lens):
        src_lens = src_lens.to(x.device)
        dst_lens = dst_lens.to(x.device)
        dst_max_len = int(dst_lens.max().item())
        out = x.new_zeros(x.shape[0], dst_max_len, x.shape[2])
        for idx in range(x.shape[0]):
            src_len = int(src_lens[idx].item())
            dst_len = int(dst_lens[idx].item())
            src = x[idx:idx + 1, :src_len]
            if src_len != dst_len:
                src = interpolate_latents(src, size=dst_len)
            out[idx, :dst_len] = src[0]
        mask = lengths_to_mask(dst_lens, dst_max_len).to(x.device)
        return torch.where(mask.unsqueeze(-1), out, torch.zeros_like(out))

    def forward(self, x, conds, m_lens):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            m_lens = m_lens.to(x.device)
            bsz, seq_len, _ = x.shape
            latent_lens = torch.div(m_lens, self.unit_length, rounding_mode="floor").clamp(min=1, max=seq_len)
            valid_mask = lengths_to_mask(latent_lens, seq_len).to(x.device)
            x = torch.where(valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
            full_noise = torch.randn_like(x)
            full_noise = torch.where(valid_mask.unsqueeze(-1), full_noise, torch.zeros_like(full_noise))

            stage_id = self.scheduler.sample_stage(x.device)
            stage_lens = self._stage_lens(latent_lens, stage_id)
            latent_stage, stage_mask = self._make_stage_latent(x, latent_lens, stage_lens)
            noise_stage, _ = self._make_stage_latent(full_noise, latent_lens, stage_lens)
            latent_stage = latent_stage.float()
            noise_stage = torch.where(stage_mask.unsqueeze(-1), noise_stage, torch.zeros_like(noise_stage))

            if stage_id == 0:
                up_latent = torch.zeros_like(latent_stage)
            else:
                prev_lens = self._stage_lens(latent_lens, stage_id - 1)
                prev_stage, _ = self._make_stage_latent(x, latent_lens, prev_lens)
                up_latent = self._upsample_stage(prev_stage.float(), prev_lens, stage_lens)

            start_sigma, end_sigma = self.scheduler.stage_sigmas()[stage_id]
            x0 = start_sigma * noise_stage + (1.0 - start_sigma) * up_latent.float()
            x1 = end_sigma * noise_stage + (1.0 - end_sigma) * latent_stage
            time_steps, ratios = self.scheduler.sample_training_timesteps(stage_id, bsz, x.device)
            ratio_view = ratios.view(bsz, 1, 1)
            xt = ratio_view * x0 + (1.0 - ratio_view) * x1
            target = (x1 - x0).float()
            xt = torch.where(stage_mask.unsqueeze(-1), xt.float(), torch.zeros_like(xt).float())
            target = torch.where(stage_mask.unsqueeze(-1), target, torch.zeros_like(target))

            pred = self.model(xt, time_steps, conds, stage_lens, scale_value=self.scales[stage_id]).float()
            pred_finite = torch.isfinite(pred).all()
            target_finite = torch.isfinite(target).all()
            if not bool(pred_finite and target_finite):
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            sq_error = (pred - target).pow(2).mean(dim=-1)
            # Match the full velocity field on valid sequence positions, normalized per sample
            # so shorter motions do not get underweighted by padding or batch-level token counts.
            per_sample_loss = (sq_error * stage_mask.float()).sum(dim=1) / stage_lens.float().clamp_min(1.0)
            raw_stage_loss = per_sample_loss.mean()
            loss_finite = torch.isfinite(raw_stage_loss.detach())
            stage_loss = torch.nan_to_num(raw_stage_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss_dict = {
                "loss": stage_loss,
                f"loss_stage_{stage_id}": stage_loss.detach(),
                "pred_finite": pred_finite.detach(),
                "target_finite": target_finite.detach(),
                "loss_finite": loss_finite,
            }
            return stage_loss, loss_dict

    def _denoise_stage(self, x, labels, latent_lens, stage_mask, stage_id, steps, cfg, cfg_interval):
        time_steps, sigmas = self.scheduler.set_timesteps(steps, stage_id, x.device)
        scale_value = self.scales[stage_id]
        for i in range(len(sigmas) - 1):
            t = time_steps[i].expand(x.shape[0])
            pred = self.model(x, t, labels, latent_lens, force_mask=False, scale_value=scale_value)
            if cfg == 1.0:
                guided = pred
            else:
                uncond = self.model(x, t, labels, latent_lens, force_mask=True, scale_value=scale_value)
                guided = uncond + cfg * (pred - uncond)
                if sigmas[i] < cfg_interval[0] or sigmas[i] > cfg_interval[1]:
                    guided = pred
            x = x + (sigmas[i] - sigmas[i + 1]) * guided
            x = torch.where(stage_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x

    @torch.no_grad()
    def generate(self, bsz, latent_lens, labels=None, cfg=1.0, device=None, steps=None,
                 temperature=1.0, cfg_interval=(0.0, 1.0)):
        device = device or latent_lens.device
        steps = steps or self.sample_steps
        latent_lens = latent_lens.to(device).long().clamp(min=1, max=self.seq_len)
        max_len = int(latent_lens.max().item())
        if labels is None:
            labels = [""] * bsz
        full_noise = torch.randn(bsz, max_len, self.token_embed_dim, device=device) * temperature
        full_mask = lengths_to_mask(latent_lens, max_len).to(device)
        full_noise = torch.where(full_mask.unsqueeze(-1), full_noise, torch.zeros_like(full_noise))
        latents = None
        prev_noise = None
        for stage_id, _scale in enumerate(self.scales):
            stage_lens = self._stage_lens(latent_lens, stage_id)
            stage_len = int(stage_lens.max().item())
            stage_mask = lengths_to_mask(stage_lens, stage_len).to(device)
            noise, _ = self._make_stage_latent(full_noise, latent_lens, stage_lens)
            noise = torch.where(stage_mask.unsqueeze(-1), noise, torch.zeros_like(noise))
            start_sigma, end_sigma = self.scheduler.stage_sigmas()[stage_id]

            if stage_id == 0:
                x_stage = noise
            else:
                prev_lens = self._stage_lens(latent_lens, stage_id - 1)
                prev_end_sigma = self.scheduler.end_sigmas[stage_id - 1]
                denoised = latents - prev_end_sigma * prev_noise
                up_latent = self._upsample_stage(denoised, prev_lens, stage_lens)
                correction = (1.0 - start_sigma) / max(1.0 - prev_end_sigma, 1e-8)
                x_stage = correction * up_latent + start_sigma * noise

            stage_steps = self.scheduler.timesteps_per_stage(stage_id, steps)
            x_stage = self._denoise_stage(
                x_stage, labels, stage_lens, stage_mask, stage_id, stage_steps, cfg, cfg_interval
            )
            latents = torch.where(stage_mask.unsqueeze(-1), x_stage, torch.zeros_like(x_stage))
            prev_noise = noise

        x = latents
        if x.shape[1] != max_len:
            x = interpolate_latents(x, size=max_len)
        mask = lengths_to_mask(latent_lens.clamp(max=max_len), max_len).to(device)
        return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

    @torch.no_grad()
    def sample_tokens(self, bsz, m_lens, cfg=1.0, cfg_schedule="linear", labels=None,
                      temperature=1.0, device=None, acc_ratio=1, cfg_interval=(0.0, 1.0)):
        device = device or m_lens.device
        latent_lens = torch.div(m_lens, self.unit_length, rounding_mode="floor").to(device)
        steps = max(1, int(self.sample_steps if acc_ratio <= 1 else self.seq_len // acc_ratio))
        return self.generate(bsz, latent_lens, labels=labels, cfg=cfg, device=device,
                             steps=steps, temperature=temperature, cfg_interval=cfg_interval)

    def train_forward(self, x, conds, m_lens):
        loss, loss_dict = self.forward(x, conds, m_lens)
        return loss_dict


def hiflow_tiny():
    return partial(HiFlowSAE, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=4, mlp_ratio=2)


def hiflow_base():
    return partial(HiFlowSAE, decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12, mlp_ratio=4)


def hiflow_large():
    return partial(HiFlowSAE, decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16, mlp_ratio=4)


def hiflow_huge():
    return partial(HiFlowSAE, decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16, mlp_ratio=4)
