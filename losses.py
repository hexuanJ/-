"""
Modern loss functions for Re-ID:
  - CircleLoss  (CVPR 2020)
  - SubCenterArcFace  (ECCV 2020)
  - AdaFace  (CVPR 2022)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Used to mask out irrelevant logits in log-sum-exp operations
_LARGE_NEG = -1e9


class CircleLoss(nn.Module):
    """Circle Loss (CVPR 2020).

    Unifies classification-based and pair-based metric learning.
    Reference: https://arxiv.org/abs/2002.10857
    """

    def __init__(self, in_features, num_classes, s=256, m=0.25):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels):
        feat = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        logits = F.linear(feat, w)  # (B, C)

        # Positive / negative separation
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Scale factors
        ap = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.0)  # alpha_p
        an = torch.clamp_min(logits.detach() + self.m, min=0.0)       # alpha_n

        delta_p = 1 - self.m
        delta_n = self.m

        pos_logits = self.s * ap * (logits - delta_p)
        neg_logits = self.s * an * (logits - delta_n)

        # log-sum-exp loss
        pos_loss = torch.logsumexp(pos_logits * one_hot + (1 - one_hot) * _LARGE_NEG, dim=1)
        neg_loss = torch.logsumexp(neg_logits * (1 - one_hot) + one_hot * _LARGE_NEG, dim=1)

        loss = F.softplus(neg_loss + pos_loss).mean()
        return loss


class SubCenterArcFace(nn.Module):
    """Sub-Center ArcFace (ECCV 2020).

    Each class has K sub-centers; reduces intra-class variation.
    Reference: https://arxiv.org/abs/2004.01159
    """

    def __init__(self, in_features, num_classes, s=64.0, m=0.50, K=3):
        super().__init__()
        self.s = s
        self.m = m
        self.K = K
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes * K, in_features)
        )
        nn.init.xavier_uniform_(self.weight)
        self.num_classes = num_classes

    def forward(self, features, labels):
        feat = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cosine_all = F.linear(feat, w)  # (B, C*K)
        cosine_all = cosine_all.view(-1, self.num_classes, self.K)
        # Take max cosine over sub-centers
        cosine, _ = cosine_all.max(dim=-1)  # (B, C)

        # ArcFace margin
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        target_logits = torch.cos(theta + self.m)
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return F.cross_entropy(output, labels)


class AdaFace(nn.Module):
    """AdaFace (CVPR 2022).

    Adapts the ArcFace margin based on feature norm (proxy for image quality).
    Reference: https://arxiv.org/abs/2204.00964
    """

    def __init__(self, in_features, num_classes, s=64.0, m=0.4,
                 h=0.333, t_alpha=0.01):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.t_alpha = t_alpha
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # Running statistics for feature norm
        self.register_buffer('batch_mean', torch.ones(1))
        self.register_buffer('batch_std', torch.ones(1))

    def forward(self, features, labels):
        # Compute feature norm and normalise
        norm = torch.norm(features, 2, 1, keepdim=True)
        feat = features / (norm + 1e-8)

        # Update running stats
        if self.training:
            with torch.no_grad():
                safe_norms = norm.detach().float()
                mean = safe_norms.mean()
                std = safe_norms.std() + 1e-8
                self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * mean
                self.batch_std = (1 - self.t_alpha) * self.batch_std + self.t_alpha * std

        # Adaptive margin
        margin_scaler = (norm.detach() - self.batch_mean) / (self.batch_std + 1e-8)
        margin_scaler = margin_scaler.clamp(-1, 1) * self.h
        # g_angular: make margin smaller for low-quality images
        m_arc = self.m + margin_scaler.squeeze(1)

        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(feat, w)
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply per-sample margin to positive class
        target_logits = torch.cos(theta + m_arc.unsqueeze(1) * one_hot)
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        return F.cross_entropy(output, labels)
