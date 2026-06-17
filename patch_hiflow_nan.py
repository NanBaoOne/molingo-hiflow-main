from pathlib import Path


TRANSFORMER_OLD = """        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
"""

TRANSFORMER_NEW = """        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
"""

TRANSFORMER_OLD_2 = """            cos, sin = rope_frequencies(position_ids, self.head_dim)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        attn_mask = None
"""

TRANSFORMER_NEW_2 = """            cos, sin = rope_frequencies(position_ids, self.head_dim)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_mask = None
"""

HIFLOW_OLD = """    def forward(self, x, conds, m_lens):
        m_lens = m_lens.to(x.device)
        bsz, seq_len, _ = x.shape
        latent_lens = torch.div(m_lens, self.unit_length, rounding_mode=\"floor\").clamp(min=1, max=seq_len)
        valid_mask = lengths_to_mask(latent_lens, seq_len).to(x.device)
        x = torch.where(valid_mask.unsqueeze(-1), x, torch.zeros_like(x))

        stage_id = self.scheduler.sample_stage(x.device)
        stage_lens = self._stage_lens(latent_lens, stage_id)
        latent_stage, stage_mask = self._make_stage_latent(x, latent_lens, stage_lens)
        noise_stage = torch.randn_like(latent_stage)
        noise_stage = torch.where(stage_mask.unsqueeze(-1), noise_stage, torch.zeros_like(noise_stage))

        if stage_id == 0:
            up_latent = torch.zeros_like(latent_stage)
        else:
            prev_lens = self._stage_lens(latent_lens, stage_id - 1)
            prev_stage, _ = self._make_stage_latent(x, latent_lens, prev_lens)
            up_latent = self._upsample_stage(prev_stage, prev_lens, stage_lens)

        start_sigma, end_sigma = self.scheduler.stage_sigmas()[stage_id]
        x0 = start_sigma * noise_stage + (1.0 - start_sigma) * up_latent
        x1 = end_sigma * noise_stage + (1.0 - end_sigma) * latent_stage
        ratio = self.scheduler.sample_ratio(bsz, x.device)
        ratio_view = ratio.view(bsz, 1, 1)
        xt = ratio_view * x0 + (1.0 - ratio_view) * x1
        target = x1 - x0
        xt = torch.where(stage_mask.unsqueeze(-1), xt, torch.zeros_like(xt))
        target = torch.where(stage_mask.unsqueeze(-1), target, torch.zeros_like(target))

        pred = self.model(xt, ratio, conds, stage_lens)
        stage_loss = F.mse_loss(pred[stage_mask], target[stage_mask])
        loss_dict = {\"loss\": stage_loss, f\"loss_stage_{stage_id}\": stage_loss.detach()}
        return stage_loss, loss_dict
"""

HIFLOW_NEW = """    def forward(self, x, conds, m_lens):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            m_lens = m_lens.to(x.device)
            bsz, seq_len, _ = x.shape
            latent_lens = torch.div(m_lens, self.unit_length, rounding_mode=\"floor\").clamp(min=1, max=seq_len)
            valid_mask = lengths_to_mask(latent_lens, seq_len).to(x.device)
            x = torch.where(valid_mask.unsqueeze(-1), x, torch.zeros_like(x))

            stage_id = self.scheduler.sample_stage(x.device)
            stage_lens = self._stage_lens(latent_lens, stage_id)
            latent_stage, stage_mask = self._make_stage_latent(x, latent_lens, stage_lens)
            latent_stage = latent_stage.float()
            noise_stage = torch.randn_like(latent_stage)
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
            ratio = self.scheduler.sample_ratio(bsz, x.device).float()
            ratio_view = ratio.view(bsz, 1, 1)
            xt = ratio_view * x0 + (1.0 - ratio_view) * x1
            target = (x1 - x0).float()
            xt = torch.where(stage_mask.unsqueeze(-1), xt.float(), torch.zeros_like(xt).float())
            target = torch.where(stage_mask.unsqueeze(-1), target, torch.zeros_like(target))

            pred = self.model(xt, ratio, conds, stage_lens).float()
            pred_finite = torch.isfinite(pred).all()
            target_finite = torch.isfinite(target).all()
            if not bool(pred_finite and target_finite):
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            raw_stage_loss = F.mse_loss(pred[stage_mask], target[stage_mask])
            loss_finite = torch.isfinite(raw_stage_loss.detach())
            stage_loss = torch.nan_to_num(raw_stage_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss_dict = {
                \"loss\": stage_loss,
                f\"loss_stage_{stage_id}\": stage_loss.detach(),
                \"pred_finite\": pred_finite.detach(),
                \"target_finite\": target_finite.detach(),
                \"loss_finite\": loss_finite,
            }
            return stage_loss, loss_dict
"""


def find_mogen_root():
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd / "mogen-hiflow", cwd.parent / "mogen-hiflow"]
    for candidate in candidates:
        if (candidate / "models" / "hiflow" / "hiflow_sae.py").is_file():
            return candidate
    raise SystemExit(
        "Could not find mogen-hiflow/models/hiflow. Run this script from molingo-hiflow "
        "or mogen-hiflow, or upload clean hiflow_sae.py and transformer.py first."
    )


def replace_once(path, old, new, marker):
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"already patched: {path.name} ({marker})")
        return False
    if old not in text:
        raise SystemExit(
            f"Patch failed for {path} ({marker}). The file does not match the expected clean "
            "server version. Upload clean hiflow_sae.py and transformer.py, then rerun."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path.name} ({marker})")
    return True


def main():
    root = find_mogen_root()
    hiflow = root / "models" / "hiflow" / "hiflow_sae.py"
    transformer = root / "models" / "hiflow" / "transformer.py"

    changed = False
    changed |= replace_once(transformer, TRANSFORMER_OLD, TRANSFORMER_NEW, "qk norm init")
    changed |= replace_once(transformer, TRANSFORMER_OLD_2, TRANSFORMER_NEW_2, "qk norm forward")
    changed |= replace_once(hiflow, HIFLOW_OLD, HIFLOW_NEW, "fp32 flow and finite loss guard")
    print("HiFlow NaN patch complete." if changed else "HiFlow NaN patch was already applied.")


if __name__ == "__main__":
    main()
