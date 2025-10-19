import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math


def categorical_cross_entropy(logits, labels, from_logits=True):
    if from_logits:
        return -(F.log_softmax(logits, dim=1) * labels).sum(dim=1).mean()
    else:
        return -(torch.log(logits) * labels).sum(dim=1).mean()

def compute_softmax_probs_for_subclusters(logits, num_classes, num_subclusters):
    out = F.softmax(logits - torch.max(logits), dim=1)
    out = torch.reshape(out, (-1, num_classes, num_subclusters))
    return torch.sum(out, dim=2)


class AdaProj(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, subspace_dim: int = 1, trainable: bool = True,
                 eps: float = 1e-12, scale_param=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.trainable = trainable
        self.eps = eps
        self.scale_param = scale_param
        self.W = nn.Parameter(data=torch.Tensor(self.num_classes, self.subspace_dim, self.emb_dim),
                              requires_grad=self.trainable)
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # manual update step
        nn.init.constant_(self.s, np.maximum(np.sqrt(2.0) * np.log(self.num_classes - 1.0), 0.5))
        self.pi = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(self.W, p=2.0, dim=2)  # num_classes x subspace_dim x emb_dim
        logits = torch.tensordot(x, W, dims=[[1], [2]])  # batchsize x num_classes x subspace_dim
        x_proj = (torch.unsqueeze(logits, dim=3) * torch.unsqueeze(W, dim=0)).sum(
            dim=2)  # batchsize x num_classes x emb_dim
        x_proj = F.normalize(x_proj, p=2.0, dim=2)
        logits = (torch.unsqueeze(x, dim=1) * x_proj).sum(dim=2)  # batchsize x num_classes
        if labels is None or not self.scale_param:
            return logits
        if self.training:
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels):
        theta = torch.acos(torch.clamp(logits, min=-1.0 + self.eps, max=1.0 - self.eps))
        max_logits = torch.max(self.s * logits)
        B_avg = torch.where(labels < 1, torch.exp(self.s * logits - max_logits),
                            torch.exp(torch.zeros_like(self.s * logits) - max_logits))
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(labels * theta, dim=1)  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp((max_logits + torch.log(B_avg)) / torch.cos(torch.minimum(self.pi / 4, theta_med)),
                        min=self.eps), requires_grad=False)


class SCAdaCos(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, num_subclusters: int = 16, trainable: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.num_subclusters = num_subclusters
        self.trainable = trainable
        self.W = nn.Parameter(data=torch.Tensor(self.num_classes * self.num_subclusters, self.emb_dim),
                              requires_grad=self.trainable)
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # manual update step
        nn.init.constant_(self.s, np.maximum(np.sqrt(2.0) * np.log(self.num_classes * self.num_subclusters - 1.0), 0.5))
        self.pi = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(self.W, p=2.0, dim=1)  # num_classes x emb_dim
        logits = torch.tensordot(x, W, dims=[[1], [1]])  # batchsize x num_classes
        if labels is None:
            return logits
        with torch.no_grad():
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels, eps=1e-12):
        theta = torch.acos(torch.clamp(logits, min=-1.0 + eps, max=1.0 - eps))
        max_logits = torch.max(self.s * logits)
        labels = torch.repeat_interleave(labels, repeats=self.num_subclusters, dim=1)
        B_avg = torch.exp(self.s * logits - max_logits)  # diverges without using mixup
        # B_avg = torch.where(labels<eps, torch.exp(self.s*logits-max_logits), torch.exp(torch.zeros_like(self.s*logits)-max_logits))
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(labels * theta, dim=1)  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp((max_logits + torch.log(B_avg)) / torch.cos(torch.minimum(self.pi / 4, theta_med)), min=eps),
            requires_grad=False)

    def compute_softmax_probs(self, logits):
        out = F.softmax(logits - torch.max(logits), dim=1)
        out = torch.reshape(out, (-1, self.num_classes, self.num_subclusters))
        return torch.sum(out, dim=2)


class AdaCos(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, trainable: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.trainable = trainable
        self.W = nn.Parameter(data=torch.Tensor(self.num_classes, self.emb_dim), requires_grad=self.trainable)
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # manual update step
        nn.init.constant_(self.s, np.maximum(np.sqrt(2.0) * np.log(self.num_classes - 1.0), 0.5))
        self.pi = nn.Parameter(data=torch.Tensor(1), requires_grad=False)  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(self.W, p=2.0, dim=1)  # num_classes x emb_dim
        logits = torch.tensordot(x, W, dims=[[1], [1]])  # batchsize x num_classes
        if labels is None:
            return logits
        with torch.no_grad():
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels, eps=1e-12):
        theta = torch.acos(torch.clamp(logits, min=-1.0 + eps, max=1.0 - eps))
        max_logits = torch.max(self.s * logits)
        B_avg = torch.where(labels < 1, torch.exp(self.s * logits - max_logits),
                            torch.exp(torch.zeros_like(self.s * logits) - max_logits))
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(labels * theta, dim=1)  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp((max_logits + torch.log(B_avg)) / (torch.cos(torch.minimum(self.pi / 4, theta_med)) + eps),
                        min=eps), requires_grad=False)