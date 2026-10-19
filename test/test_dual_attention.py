import torch
from torch_geometric.utils import scatter

from models.rgat_deep import DeepRGAT, DualAttentionConv


def _star_graph():
    # three neighbours (0, 1, 2) -> centre node 3, all with relation 0
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
    edge_type = torch.zeros(3, dtype=torch.long)
    return edge_index, edge_type


def _attention_weights(conv, x, edge_index, edge_type, rel_emb, recency):
    weights = {}

    def capture(module, inputs, output):
        weights["alpha"] = output

    # recover the normalised weights by sending ones as messages
    with torch.no_grad():
        conv.V.fill_(0.0)
        for r in range(conv.num_rel):
            conv.V[r, 0, :] = 1.0
        x = x.clone()
        x[:, 0] = 1.0
        handle = conv.register_message_forward_hook(capture)
        conv.eval()
        conv(x, edge_index, edge_type, rel_emb, recency)
        handle.remove()
    return weights["alpha"].view(edge_index.size(1), conv.heads, conv.head_dim)[:, :, 0]


def test_recent_edges_get_larger_weight_when_alpha_is_equal():
    edge_index, edge_type = _star_graph()
    conv = DualAttentionConv(4, 2, num_rel=1, rel_dim=4, heads=2, dropout=0.0,
                             use_alpha=False)
    x = torch.randn(4, 4)
    recency = torch.tensor([1.0, 0.5 ** (1 / 3), 0.5 ** (2 / 3)])  # visits k = 0, 1, 2
    w = _attention_weights(conv, x, edge_index, edge_type, torch.randn(1, 4), recency)
    assert torch.all(w[0] > w[1]) and torch.all(w[1] > w[2])


def test_weights_are_finite_nonnegative_and_sum_to_one_per_relation():
    edge_index = torch.tensor([[0, 1, 2, 0, 1], [3, 3, 3, 3, 3]])
    edge_type = torch.tensor([0, 0, 0, 1, 1])
    conv = DualAttentionConv(4, 2, num_rel=2, rel_dim=4, heads=2, dropout=0.0)
    w = _attention_weights(conv, torch.randn(4, 4), edge_index, edge_type,
                           torch.randn(2, 4), torch.rand(5))
    assert torch.isfinite(w).all() and (w >= 0).all()
    sums = scatter(w, edge_type, dim=0, reduce="sum")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gradients_are_finite():
    edge_index, edge_type = _star_graph()
    model = DeepRGAT(num_nodes=4, num_rel=1, hid=8, num_layers=2, heads=2)
    out = model(edge_index, edge_type, edge_weight=torch.rand(3, 1),
                batch=torch.zeros(4, dtype=torch.long), x_idx=torch.arange(4))
    out.sum().backward()
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
