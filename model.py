"""Small masked spatial GNN + causal TCN; all coordinates and outputs are metres."""
import torch
from torch import nn
from torch.nn import functional as F


def constant_velocity(x, mask):
    """Exactly x(t) + v(t) * 1 s. Reject unavailable player anchors."""
    # Dataset auditing guarantees all 22 player anchor values are observed.
    anchor = torch.where(mask[:, -1, :22], x[:, -1, :22], 0.0)
    return anchor[..., :2] + anchor[..., 2:4]


class SpatialGraph(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.neighbor = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, graphs):
        out = self.self_linear(h)
        for layer, graph in zip(self.neighbor, graphs):
            out = out + layer(torch.matmul(graph, h))
        return self.norm(h + F.gelu(out))


class CausalTCNBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.left_pad = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # [batch * nodes, channels, time]; left padding makes convolution causal.
        z = self.pointwise(F.gelu(self.depthwise(F.pad(h, (self.left_pad, 0)))))
        return self.norm((h + self.dropout(z)).transpose(1, 2)).transpose(1, 2)


class STGNN_TCN(nn.Module):
    def __init__(self, mean, std, width=32, dropout=0.1, distance_scale_m=20.0):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).clamp_min(1e-6))
        self.distance_scale_m = distance_scale_m
        # 4 normalized observations + 4 masks + 3 team types.
        self.input_projection = nn.Linear(11, width)
        self.spatial = SpatialGraph(width)
        self.temporal = nn.ModuleList(CausalTCNBlock(width, d, dropout) for d in [1, 2, 4, 8, 16])
        self.final_spatial = SpatialGraph(width)
        self.head = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def make_graphs(self, x, mask, team):
        # No missing node can send OR receive a spatial message.
        pos = torch.where(mask[..., :2], x[..., :2], 0.0)
        dist2 = (pos.unsqueeze(-2) - pos.unsqueeze(-3)).square().sum(-1)
        valid = mask[..., :2].all(-1)
        edges = valid.unsqueeze(-1) & valid.unsqueeze(-2)
        edges = edges & ~torch.eye(23, dtype=torch.bool, device=x.device)
        weights = torch.exp(-dist2 / (2 * self.distance_scale_m ** 2)) * edges
        target_team = team[:, None, :, None]
        source_team = team[:, None, None, :]
        # Teammates, opponents, and incoming ball messages have separate parameters.
        relations = [(target_team == source_team) & (source_team != 2),
                     (target_team != source_team) & (source_team != 2), source_team == 2]
        graphs = []
        for relation in relations:
            a = weights * relation
            graphs.append(a / a.sum(-1, keepdim=True).clamp_min(1e-6))
        return graphs

    def forward(self, x, mask, team):
        # Y, IDs, match IDs and future frame metadata are deliberately absent.
        z = torch.where(mask, (x - self.mean) / self.std, 0.0)
        onehot = F.one_hot(team.long(), 3).to(x.dtype)[:, None].expand(-1, x.shape[1], -1, -1)
        h = F.gelu(self.input_projection(torch.cat([z, mask.to(x.dtype), onehot], -1)))
        graphs = self.make_graphs(x, mask, team)
        h = self.spatial(h, graphs)
        b, t, n, c = h.shape
        h = h.permute(0, 2, 3, 1).reshape(b * n, c, t)
        for block in self.temporal:
            h = block(h)
        h = h[:, :, -1].reshape(b, n, c)
        h = self.final_spatial(h, [a[:, -1] for a in graphs])
        correction = self.head(torch.cat([h[:, :22], z[:, -1, :22]], -1))
        return constant_velocity(x, mask) + correction
