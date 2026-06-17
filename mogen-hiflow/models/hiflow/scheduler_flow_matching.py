import torch


class FlowMatchingScheduler:
    def __init__(self, num_steps=32):
        self.num_steps = num_steps

    def sample_time(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def add_noise(self, x1, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x1)
        view_shape = (x1.shape[0],) + (1,) * (x1.dim() - 1)
        t_view = t.view(view_shape)
        xt = (1.0 - t_view) * noise + t_view * x1
        target = x1 - noise
        return xt, target, noise

    def timesteps(self, device, steps=None):
        steps = steps or self.num_steps
        return torch.linspace(0.0, 1.0, steps + 1, device=device)

    @torch.no_grad()
    def euler(self, model_fn, noise, steps=None):
        x = noise
        ts = self.timesteps(noise.device, steps=steps)
        for i in range(len(ts) - 1):
            t = ts[i].expand(noise.shape[0])
            dt = ts[i + 1] - ts[i]
            x = x + dt * model_fn(x, t)
        return x


class PyramidFlowMatchingScheduler(FlowMatchingScheduler):
    def __init__(self, num_steps=32, scales=(0.3, 0.6, 1.0)):
        super().__init__(num_steps)
        self.scales = tuple(float(scale) for scale in scales)
        if not self.scales:
            raise ValueError("Pyramid flow requires at least one scale")
        if any(scale <= 0.0 or scale > 1.0 for scale in self.scales):
            raise ValueError("Pyramid scales must be in (0, 1]")
        if any(self.scales[i] <= self.scales[i - 1] for i in range(1, len(self.scales))):
            raise ValueError("Pyramid scales must be strictly increasing")

    def stage_sigmas(self, device=None):
        sigmas = []
        prev_scale = 0.0
        for scale in self.scales:
            sigmas.append((1.0 - prev_scale, 1.0 - scale))
            prev_scale = scale
        if device is None:
            return sigmas
        return [(torch.tensor(start, device=device), torch.tensor(end, device=device))
                for start, end in sigmas]

    def stage_weights(self, device):
        weights = torch.tensor([start - end for start, end in self.stage_sigmas()], device=device)
        return weights / weights.sum().clamp_min(1e-8)

    def sample_stage(self, device):
        return torch.multinomial(self.stage_weights(device), 1).item()

    def sample_ratio(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    @torch.no_grad()
    def euler_interval(self, model_fn, x, start_ratio=1.0, end_ratio=0.0, steps=None):
        steps = steps or self.num_steps
        ratios = torch.linspace(start_ratio, end_ratio, steps + 1, device=x.device)
        for i in range(len(ratios) - 1):
            ratio = ratios[i].expand(x.shape[0])
            dr = ratios[i] - ratios[i + 1]
            x = x + dr * model_fn(x, ratio)
        return x
