import torch
from torch import nn
import math
import numpy as np
import dgl
from dgl.nn.pytorch import GATv2Conv
from dgl.nn.pytorch.glob import GlobalAttentionPooling



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, dev='cpu'):
        super().__init__()
        layers = []
        for i, k in enumerate(kernel_sizes):
            d = dilation ** i
            pad = (k - 1) * d
            in_c = num_inputs if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            layers += [
                nn.Conv1d(in_c, out_c, k, padding=pad, dilation=d),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                Chomp1d(pad)
            ]
        self.net = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]
        self.net.to(dev)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [N, C, T]
        x = self.net(x)
        return x.permute(0, 2, 1)  # [N, T, D]

###Modal encoder

class TraceModel(nn.Module):
    def __init__(self, device='cpu', trace_hiddens=[64], trace_kernel_sizes=[3], **kwargs):
        super().__init__()
        self.net = ConvNet(2, trace_hiddens, trace_kernel_sizes, dev=device)
        self.out_dim = trace_hiddens[-1]

    def forward(self, x):
        return self.net(x)


class MetricModel(nn.Module):
    def __init__(self, metric_num, device='cpu', metric_hiddens=[64], metric_kernel_sizes=[3], **kwargs):
        super().__init__()
        self.net = ConvNet(metric_num, metric_hiddens, metric_kernel_sizes, dev=device)
        self.out_dim = metric_hiddens[-1]

    def forward(self, x):
        return self.net(x)


class LogModel(nn.Module):
    def __init__(self, event_num, out_dim):
        super().__init__()
        self.proj = nn.Linear(event_num, out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.proj(x)


### CrossModalAttention

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_modalities=3):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.key_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_modalities)])
        self.val_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_modalities)])
        self.scale = d_model ** -0.5

    def forward(self, feats):
        N, T, D = feats[0].shape
        Q = self.query.expand(N, T, D)
        scores, values = [], []
        for i, f in enumerate(feats):
            K = self.key_proj[i](f)
            V = self.val_proj[i](f)
            scores.append((Q * K).sum(-1) * self.scale)
            values.append(V)
        scores = torch.stack(scores, dim=-1)
        attn = torch.softmax(scores, dim=-1)
        fused = 0
        for i in range(len(values)):
            fused += attn[..., i:i + 1] * values[i]
        return fused


# 多模态编码 + 时间 & 空间建模
class MultiSourceEncoder(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device,
                 log_dim=64, fuse_dim=64, chunk_lenth=None, **kwargs):
        super().__init__()
        self.node_num = node_num

        self.num_heads = kwargs.get("attn_head", 4)
        d_model = fuse_dim

        self.trace_model = TraceModel(device=device, **kwargs)
        self.metric_model = MetricModel(metric_num, device=device, **kwargs)
        self.log_model = LogModel(event_num, log_dim)

        self.proj_trace = nn.Linear(self.trace_model.out_dim, d_model)
        self.proj_metric = nn.Linear(self.metric_model.out_dim, d_model)
        self.proj_log = nn.Linear(log_dim, d_model)

        self.cross_attn = CrossModalAttention(d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )


        assert d_model % self.num_heads == 0, f"fuse_dim ({d_model}) 必须能被 attn_head ({self.num_heads}) 整除"
        head_dim = d_model // self.num_heads

        self.gat_in = GATv2Conv(d_model, head_dim, num_heads=self.num_heads, allow_zero_in_degree=True)
        self.gat_out = GATv2Conv(d_model, head_dim, num_heads=self.num_heads, allow_zero_in_degree=True)

        self.dir_fuse = nn.Linear(d_model * 2, d_model)
        self.out_dim = d_model

    def forward(self, graph):
        trace = self.proj_trace(self.trace_model(graph.ndata["traces"]))
        metric = self.proj_metric(self.metric_model(graph.ndata["metrics"]))
        log = self.proj_log(self.log_model(graph.ndata["logs"]))

        T = trace.size(1)
        metric = self.align_time(metric, T)
        log = self.align_time(log, T)

        fused_seq = self.cross_attn([trace, metric, log])

        temporal_feat = self.transformer(fused_seq).mean(dim=1)

        spatial_input = fused_seq.mean(dim=1)
        g_rev = dgl.reverse(graph, copy_ndata=False)

        h_in = self.gat_in(g_rev, spatial_input).flatten(1)
        h_out = self.gat_out(graph, spatial_input).flatten(1)

        spatial_feat = self.dir_fuse(torch.cat([h_in, h_out], dim=-1))

        return temporal_feat + spatial_feat

    def align_time(self, x, T):
        if x.size(1) == T: return x
        if x.size(1) == 1: return x.repeat(1, T, 1)
        return x[:, :T, :]



class FullyConnected(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        layers = []
        for i, h in enumerate(hidden):
            layers += [nn.Linear(in_dim if i == 0 else hidden[i - 1], h), nn.ReLU()]
        layers.append(nn.Linear(hidden[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MainModel(nn.Module):
    def __init__(self, event_num, metric_num, node_num, device, alpha=0.5, **kwargs):
        super().__init__()
        self.encoder = MultiSourceEncoder(event_num, metric_num, node_num, device, **kwargs)
        self.locator = FullyConnected(self.encoder.out_dim, node_num, kwargs["locate_hiddens"])
        self.detector = FullyConnected(self.encoder.out_dim, 2, kwargs["detect_hiddens"])
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, graph, fault_index):
        h = self.encoder(graph)
        if h.dim() == 2 and h.shape[0] > fault_index.shape[0]:
            T = h.shape[0] // fault_index.shape[0]
            h = h.view(fault_index.shape[0], T, -1)
        if h.dim() == 3:
            h = h.mean(dim=1)

        locate_logits = self.locator(h)
        detect_logits = self.detector(h)
        locate_loss = self.ce(locate_logits, fault_index.to(h.device))
        detect_loss = self.ce(detect_logits, (fault_index > -1).long().to(h.device))
        loss = self.alpha * detect_loss + (1 - self.alpha) * locate_loss
        probs = torch.softmax(locate_logits, dim=-1).detach().cpu().numpy()

        return {
            "loss": loss,
            "pred_prob": probs,
            "y_pred": np.flip(probs.argsort(axis=1), axis=1),
            "y_prob": probs
        }