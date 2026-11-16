"""Thin adapters; official RVQ and both Transformers remain byte-for-byte intact."""
import bootstrap  # noqa: F401
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from models.vq.model import RVQVAE


def build_vq(config):
    opt = SimpleNamespace(mu=config['mu'], num_quantizers=config['num_quantizers'],
                          shared_codebook=False, quantize_dropout_prob=config['quantize_dropout_prob'])
    return RVQVAE(opt, input_width=272, nb_code=config['nb_code'], code_dim=config['code_dim'],
                  output_emb_width=config['code_dim'], down_t=2, stride_t=2,
                  width=config['width'], depth=config['depth'], dilation_growth_rate=3,
                  activation='relu', norm=None)


class SemanticRVQ(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vq = build_vq(config)
        self.semantic_proj = nn.Linear(1024, config['code_dim'])

    def forward(self, motion, babel=None, lengths=None, has_babel=None):
        z = self.vq.encoder(self.vq.preprocess(motion))
        # Official ResidualVQ updates its residual in place. Clone to preserve
        # the pre-quantization representation used by the semantic objective.
        with torch.autocast(device_type=motion.device.type, enabled=False):
            q, _, commit, perplexity = self.vq.quantizer(z.float().clone(), sample_codebook_temp=0.5)
        reconstruction = self.vq.decoder(q)
        semantic = z.new_zeros(())
        if babel is not None:
            semantic = local_alignment(z.transpose(1, 2).float(), babel.float(), lengths, has_babel,
                                       self.semantic_proj)
        return reconstruction, commit, perplexity, semantic


def local_alignment(z, babel, lengths, has_babel, projection):
    """MoLingo-style causal BABEL pooling, stride=4/history=16.

    Validity/duplicate filtering is per sample, never across sample boundaries.
    The target is projected by a learned linear head, not a new text encoder.
    """
    if not has_babel.any():
        return z.sum() * 0
    positions = torch.arange(z.shape[1], device=z.device)
    ends = (positions + 1) * 4
    starts = (positions * 4 - 16).clamp_min(0)
    sums = F.pad(babel.cumsum(dim=1), (0, 0, 1, 0))
    pooled = (sums[:, ends] - sums[:, starts]) / (ends - starts)[None, :, None]
    targets = projection(pooled)
    valid = (positions[None] < (lengths // 4)[:, None]) & has_babel[:, None]
    # Compare normalized source semantics, within each sequence only.
    # A repeated action does not acquire a new boundary at the next sample.
    normalized = F.normalize(pooled, dim=-1)
    unique = torch.ones_like(valid)
    unique[:, 1:] = (normalized[:, 1:] * normalized[:, :-1]).sum(-1) < 0.995
    valid &= unique
    if not valid.any():
        return z.sum() * 0
    return (1 - F.cosine_similarity(z[valid], targets[valid], dim=-1)).mean()


def build_transformer(config, stage, device):
    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
    opt = SimpleNamespace(num_tokens=config['nb_code'], num_quantizers=config['num_quantizers'], device=device)
    kwargs = dict(code_dim=config['code_dim'], cond_mode='text', latent_dim=config['latent_dim'],
                  ff_size=config['ff_size'], num_layers=config['layers'], num_heads=config['heads'],
                  dropout=config['dropout'], clip_dim=512, cond_drop_prob=0.1,
                  clip_version='ViT-B/32', opt=opt)
    if stage == 'mask':
        model = MaskTransformer(**kwargs)
    else:
        model = ResidualTransformer(**kwargs, shared_codebook=False, share_weight=True)
    return model.to(device)


def pack_ema(vq):
    # Official EMA's Python attributes are absent from state_dict: preserve them
    # to prevent codebook re-initialization on resume.
    return [dict(init=x.init, code_sum=x.code_sum, code_count=x.code_count) for x in vq.quantizer.layers]


def restore_ema(vq, states, device):
    for layer, state in zip(vq.quantizer.layers, states):
        layer.init = state['init']
        for key in ('code_sum', 'code_count'):
            value = state[key]
            setattr(layer, key, value.to(device) if value is not None else None)
