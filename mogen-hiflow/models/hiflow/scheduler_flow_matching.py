import math

import numpy as np
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
    def __init__(self, num_steps=32, scales=(0.3, 0.6, 1.0), num_train_timesteps=1000,
                 shift=1.0, gamma=1.0):
        super().__init__(num_steps)
        self.scales = tuple(float(scale) for scale in scales)
        if not self.scales:
            raise ValueError("Pyramid flow requires at least one scale")
        if any(scale <= 0.0 or scale > 1.0 for scale in self.scales):
            raise ValueError("Pyramid scales must be in (0, 1]")
        if any(self.scales[i] <= self.scales[i - 1] for i in range(1, len(self.scales))):
            raise ValueError("Pyramid scales must be strictly increasing")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.gamma = float(gamma)
        self.stage_range = [0.0] + list(self.scales)
        self._init_sigmas_for_each_stage()

    def _shift_sigma(self, sigma):
        if self.shift == 1.0:
            return sigma
        return self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)

    def _init_sigmas_for_each_stage(self):
        training_steps = self.num_train_timesteps
        timesteps = np.linspace(1, training_steps, training_steps, dtype=np.float64)[::-1].copy()
        sigmas = self._shift_sigma(timesteps / training_steps)

        self.timesteps = torch.from_numpy(sigmas * training_steps).float()
        self.sigmas = torch.from_numpy(sigmas).float()

        self.start_sigmas = []
        self.end_sigmas = []
        self.ori_start_sigmas = []
        self.timestep_ratios = {}
        self.timesteps_per_stage_dict = {}
        self.sigmas_per_stage_dict = {}

        stage_distance = []

        for stage_id in range(len(self.scales)):
            start_indice = max(int(self.stage_range[stage_id] * training_steps), 0)
            end_indice = min(int(self.stage_range[stage_id + 1] * training_steps), training_steps)
            start_sigma = float(sigmas[start_indice])
            end_sigma = float(sigmas[end_indice]) if end_indice < training_steps else 0.0
            self.ori_start_sigmas.append(start_sigma)

            if stage_id > 0:
                ori_sigma = 1.0 - start_sigma
                corrected_sigma = (
                    (1.0 / (math.sqrt(1.0 + (1.0 / self.gamma)) * (1.0 - ori_sigma) + ori_sigma))
                    * ori_sigma
                )
                start_sigma = 1.0 - corrected_sigma

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas.append(start_sigma)
            self.end_sigmas.append(end_sigma)

        total_distance = sum(stage_distance)
        for stage_id in range(len(self.scales)):
            if stage_id == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:stage_id]) / total_distance
            if stage_id == len(self.scales) - 1:
                end_ratio = 1.0
            else:
                end_ratio = sum(stage_distance[:stage_id + 1]) / total_distance

            self.timestep_ratios[stage_id] = (start_ratio, end_ratio)

        for stage_id in range(len(self.scales)):
            ratio_start, ratio_end = self.timestep_ratios[stage_id]
            timestep_max = self.timesteps[int(ratio_start * training_steps)].item()
            timestep_min = self.timesteps[min(int(ratio_end * training_steps), training_steps - 1)].item()
            stage_timesteps = np.linspace(
                timestep_max, timestep_min, training_steps + 1, dtype=np.float64,
            )[:-1]
            stage_sigmas = np.linspace(1.0, 0.0, training_steps + 1, dtype=np.float64)[:-1]

            self.timesteps_per_stage_dict[stage_id] = torch.from_numpy(
                stage_timesteps
            ).float()
            self.sigmas_per_stage_dict[stage_id] = torch.from_numpy(stage_sigmas).float()

    def stage_sigmas(self, device=None):
        if device is None:
            return list(zip(self.start_sigmas, self.end_sigmas))
        return [(torch.tensor(start, device=device), torch.tensor(end, device=device))
                for start, end in zip(self.start_sigmas, self.end_sigmas)]

    def stage_weights(self, device):
        widths = [max(start - end, 1e-8) ** self.gamma for start, end in self.stage_sigmas()]
        weights = torch.tensor(widths, dtype=torch.float32, device=device)
        return weights / weights.sum().clamp_min(1e-8)

    def sample_stage(self, device):
        return torch.multinomial(self.stage_weights(device), 1).item()

    def timesteps_per_stage(self, stage_id, num_inference_steps=None):
        num_inference_steps = int(num_inference_steps or self.num_steps)
        stage_distance = max(self.start_sigmas[stage_id] - self.end_sigmas[stage_id], 0.0)
        return max(1, int(round(num_inference_steps * stage_distance)))

    def sigmas_per_stage(self, stage_id, num_inference_steps=None, device=None):
        steps = self.timesteps_per_stage(stage_id, num_inference_steps)
        sigmas = torch.linspace(1.0, 0.0, steps, device=device)
        return torch.cat([sigmas, sigmas.new_zeros(1)])

    def sample_training_timesteps(self, stage_id, batch_size, device):
        indices = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        time_steps = self.timesteps_per_stage_dict[stage_id].to(device)[indices]
        ratios = self.sigmas_per_stage_dict[stage_id].to(device)[indices]
        return time_steps.float(), ratios.float()

    def sample_ratio(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def set_timesteps(self, num_inference_steps, stage_id, device):
        steps = int(num_inference_steps)
        stage_timesteps = self.timesteps_per_stage_dict[stage_id]
        timesteps = torch.linspace(
            stage_timesteps[0].item(),
            stage_timesteps[-1].item(),
            steps,
            device=device,
        )
        stage_sigmas = self.sigmas_per_stage_dict[stage_id]
        sigmas = torch.linspace(
            stage_sigmas[0].item(),
            stage_sigmas[-1].item(),
            steps,
            device=device,
        )
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        self.timesteps = timesteps
        self.sigmas = sigmas
        return timesteps, sigmas

    @torch.no_grad()
    def euler_interval(self, model_fn, x, start_ratio=1.0, end_ratio=0.0, steps=None):
        steps = steps or self.num_steps
        ratios = torch.linspace(start_ratio, end_ratio, steps + 1, device=x.device)
        for i in range(len(ratios) - 1):
            ratio = ratios[i].expand(x.shape[0])
            dr = ratios[i] - ratios[i + 1]
            x = x + dr * model_fn(x, ratio)
        return x
