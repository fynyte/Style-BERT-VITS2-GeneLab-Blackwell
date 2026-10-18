"""Text-side, multimodal prosody prediction for JP-Extra fine-tuning.

One mixture component is selected for an entire utterance. Averaging component
means at inference would undo the reason for using a mixture in the first place.
The old checkpoint's tensor names/shapes are never changed.
"""
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ProsodyAdapter(nn.Module):
    def __init__(self, hidden_channels: int, inter_channels: int, components: int = 4):
        super().__init__()
        if components < 1:
            raise ValueError("prosody_components must be positive")
        self.components = components
        self.style = nn.Linear(256, hidden_channels)
        self.context = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1), nn.SiLU(),
        )
        self.mixture = nn.Linear(hidden_channels, components)
        self.statistics = nn.Conv1d(hidden_channels, components * 6, 1)
        self.voicing = nn.Conv1d(hidden_channels, 1, 1)
        self.prior_residual = nn.Conv1d(3, inter_channels, 1, bias=False)
        self.duration_style = nn.Linear(256, hidden_channels)
        self.duration_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.prior_residual.weight)
        nn.init.zeros_(self.duration_style.weight)
        nn.init.zeros_(self.duration_style.bias)

    def predict(self, hidden, mask, style_vec):
        h = self.context((hidden + self.style(style_vec).unsqueeze(-1)) * mask) * mask
        pooled = h.sum(-1) / mask.sum(-1).clamp_min(1)
        logits = self.mixture(pooled)
        # Keep distribution arithmetic in FP32, including under FP16 autocast.
        stats = self.statistics(h).float().view(h.size(0), self.components, 6, h.size(-1))
        means, log_scales = stats[:, :, :3], stats[:, :, 3:].clamp(-3.0, 2.0)
        return logits.float(), means, log_scales, self.voicing(h).float()

    @staticmethod
    def targets(frame_features, attn, durations):
        """Pool frame F0 only over voiced frames; retain unvoiced/silent phones.

        frame_features: [B, 3, frames] = normalized log-F0, log-RMS, voiced.
        attn: detached MAS alignment [B, 1, frames, phones].
        """
        path = attn[:, 0].float()
        frames = frame_features.float()
        count = durations.float().clamp_min(1)
        voiced_frames = frames[:, 2:3]
        voiced_count = torch.matmul(voiced_frames, path)
        pitch = torch.matmul(frames[:, :1] * voiced_frames, path) / voiced_count.clamp_min(1)
        energy = torch.matmul(frames[:, 1:2], path) / count
        voiced = voiced_count / count
        # Fixed units, not utterance normalization: preserve expressive ranges.
        log_duration = (torch.log(count) - math.log(4.0)) / 1.5
        return torch.cat((pitch, energy, log_duration), 1), voiced

    @staticmethod
    def component_scores(prediction, target, voiced, mask):
        logits, means, log_scales, _ = prediction
        weight = torch.cat((voiced, torch.ones_like(voiced), torch.ones_like(voiced)), 1)
        weight = weight * mask.float()
        log_prob = -0.5 * ((target[:, None] - means) * torch.exp(-log_scales)).square()
        log_prob = log_prob - log_scales - 0.5 * math.log(2 * math.pi)
        # Sequence likelihood, with a single component shared by all phones.
        scores = (log_prob * weight[:, None]).sum((2, 3))
        return scores + F.log_softmax(logits, -1), weight.sum((1, 2)).clamp_min(1)

    def training_loss_and_features(self, prediction, target, voiced, mask):
        scores, count = self.component_scores(prediction, target, voiced, mask)
        nll = (-torch.logsumexp(scores, -1) / count).mean()
        voiced_loss = (F.binary_cross_entropy_with_logits(
            prediction[3], voiced, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
        # The posterior chooses a component, rather than blending different acts.
        index = scores.detach().argmax(-1)
        selected = self.select(prediction[1], index)
        # Expose the acoustic model to predicted features during training too.
        teacher = torch.rand(target.size(0), 1, 1, device=target.device) < 0.5
        acoustic = torch.where(teacher, target[:, :2], selected[:, :2])
        voice = torch.where(teacher, voiced, prediction[3].sigmoid())
        features = torch.cat((acoustic[:, :1] * voice, acoustic[:, 1:2], voice), 1) * mask
        return nll + voiced_loss, features, selected

    @staticmethod
    def select(means, index):
        batch = torch.arange(means.size(0), device=means.device)
        return means[batch, index]

    def infer_features(self, prediction, mask, noise_scale: float = 0.0,
                       component: Optional[int] = None):
        if noise_scale < 0 or not math.isfinite(noise_scale):
            raise ValueError("prosody_noise_scale must be finite and nonnegative")
        logits, means, _, voiced_logits = prediction
        if component is not None:
            if not 0 <= component < self.components:
                raise ValueError(f"prosody_component must be in [0, {self.components})")
            index = torch.full((means.size(0),), component, device=means.device, dtype=torch.long)
        elif noise_scale > 0:
            index = torch.multinomial(F.softmax(logits / max(noise_scale, 0.05), -1), 1).squeeze(1)
        else:
            index = logits.argmax(-1)
        selected = self.select(means, index)
        voiced = voiced_logits.sigmoid()
        features = torch.cat((selected[:, :1] * voiced, selected[:, 1:2], voiced), 1) * mask
        return features, selected

    def condition_duration(self, style_vec):
        # Passed AFTER the legacy predictors' detach, so rhythm can train style.
        return self.duration_style(style_vec).unsqueeze(-1)

    def correct_duration(self, logw, selected, mask):
        predicted = selected[:, 2:3] * 1.5 + math.log(4.0)
        # Bounded residual; initial output is exactly the legacy duration.
        return logw + self.duration_gate.tanh() * (predicted - logw.detach()).clamp(-2, 2) * mask
