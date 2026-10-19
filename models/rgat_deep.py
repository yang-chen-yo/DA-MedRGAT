import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax


class DualAttentionConv(MessagePassing):
    """One DA-MedRGAT layer (paper Eqs. 3-8).

    alpha_ij = LeakyReLU(a^T [W x_i || W x_j || U e_r])             (Eq. 4)
    fused    = softmax over N_i^(r) of (alpha_ij + log beta_ij,r)    (Eqs. 5-6)
    z_i      = W_0 x_i + sum_r sum_j fused_ij,r * V_r x_j            (Eqs. 7-8)

    The self-connection W_0 x_i keeps the state of nodes that have no
    incoming edges (e.g. the patient node); without it their state is zeroed
    after the first layer and the graph representation collapses.
    """

    def __init__(self, in_dim, head_dim, num_rel, rel_dim, heads, dropout,
                 use_alpha=True, negative_slope=0.2):
        super().__init__(aggr="add", node_dim=0)
        self.heads, self.head_dim, self.num_rel = heads, head_dim, num_rel
        self.use_alpha = use_alpha
        self.dropout = dropout
        self.negative_slope = negative_slope
        out_dim = heads * head_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.U = nn.Linear(rel_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, 3 * head_dim))
        self.V = nn.Parameter(torch.empty(num_rel, in_dim, out_dim))
        self.root = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.att)
        for v in self.V:  # per relation; a 3-D xavier init would use in*out as fan-in
            nn.init.xavier_uniform_(v)

    def forward(self, x, edge_index, edge_type, rel_emb, recency=None):
        h = self.W(x).view(-1, self.heads, self.head_dim)
        r = self.U(rel_emb).view(-1, self.heads, self.head_dim)
        xv = torch.einsum("ni,rio->nro", x, self.V)
        src_msg = xv[edge_index[0], edge_type]
        neigh = self.propagate(edge_index, h=h, r=r, edge_type=edge_type,
                               src_msg=src_msg, recency=recency)
        return self.root(x) + neigh

    def message(self, h_i, h_j, r, edge_type, src_msg, recency, index, size_i):
        num_edges = edge_type.size(0)
        if self.use_alpha:
            cat = torch.cat([h_i, h_j, r[edge_type]], dim=-1)
            logits = F.leaky_relu((cat * self.att).sum(-1), self.negative_slope)
        else:
            logits = h_i.new_zeros(num_edges, self.heads)
        if recency is not None:
            # Adding w equals adding log(beta) of Eq. 5: within each (i, r)
            # segment the two differ only by a constant, which softmax removes.
            logits = logits + recency.view(-1, 1)
        segment = index * self.num_rel + edge_type
        alpha = softmax(logits, segment, num_nodes=size_i * self.num_rel)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = src_msg.view(num_edges, self.heads, self.head_dim)
        return (msg * alpha.unsqueeze(-1)).view(num_edges, -1)


class DeepRGAT(nn.Module):
    """Stack of DualAttentionConv layers with mean pooling (Eqs. 9-10).

    use_alpha / use_beta switch the semantic attention and the visit-recency
    prior on or off for the ablations.
    """

    def __init__(self, num_nodes, num_rel, hid=64, num_layers=3, heads=8,
                 out_dim=1, dropout=0.15, use_alpha=True, use_beta=True):
        super().__init__()
        if hid % heads != 0:
            raise ValueError(f"hid={hid} must be divisible by heads={heads}")
        self.use_beta = use_beta
        self.dropout = dropout
        self.emb = nn.Embedding(num_nodes, hid)
        self.rel_emb = nn.Embedding(num_rel, hid)
        self.layers = nn.ModuleList(
            DualAttentionConv(hid, hid // heads, num_rel, hid, heads, dropout,
                              use_alpha=use_alpha)
            for _ in range(num_layers)
        )
        self.lin = nn.Linear(hid, out_dim)

    def forward(self, edge_index, edge_type, edge_weight=None, batch=None,
                x_idx=None):
        x = self.emb(x_idx)
        recency = edge_weight.view(-1) if (self.use_beta and edge_weight is not None) else None
        for conv in self.layers:
            x = conv(x, edge_index, edge_type, self.rel_emb.weight, recency)
            x = F.dropout(F.elu(x), p=self.dropout, training=self.training)
        return self.lin(global_mean_pool(x, batch))
