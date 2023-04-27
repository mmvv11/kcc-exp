import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


def disable_grad(module):
    for param in module.parameters():
        param.requires_grad = False


def _init_input_modules(g, ntype, hidden_dims):
    module_dict = nn.ModuleDict()
    for column, data in g.nodes[ntype].data.items():
        if column == dgl.NID:
            continue
        if data.dtype == torch.float32:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], hidden_dims)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, hidden_dims, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m
    return module_dict


class LinearProjector(nn.Module):
    """
    GCN의 projection layer를 나타냄.
    노드 피쳐들이 모두 같은 shape의 output으로 나오게 하여, aggregate를 할 수 있는 형태로 만들어야 한다.
    그리고 이결과를 코드에서는 torch.stack()으로 aggregate했다.

    """
    def __init__(self, full_graph, ntype, hidden_dims):
        super().__init__()
        self.ntype = ntype
        self.inputs = _init_input_modules(full_graph, ntype, hidden_dims)

    def forward(self, ndata):
        projections = []
        for feature, data in ndata.items():
            if feature == dgl.NID or feature.endswith('__len'):
                continue
            module = self.inputs[feature]
            result = module(data)
            projections.append(result)
        return torch.stack(projections, 1).sum(1)


class WeightedSAGEConv(nn.Module):
    """
    대부분의 계산이 이루어지는 핵심 레이어.
    projection된 입력값을 FC에 한번 더 통과시킨 후, Message passing을 통해 논문 내용대로 Conv Task를 수행한다.
    이는 PinSAGE Sampling에서 계산된 PPR score를 node feature와 행렬곱해주는 과정을 포함하며, 연산 후 다시 FC를 거쳐 normalize까지 한다.
    """
    def __init__(self, input_dims, hidden_dims, output_dims, act=F.relu):
        super().__init__()
        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.reset_parameters()
        self.dropout = nn.Dropout(0.5)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, block, h, weights):
        h_src, h_dst = h
        with block.local_scope():
            block.srcdata['nft'] = self.act(self.Q(self.dropout(h_src)))
            block.edata['w'] = weights.float()
            block.update_all(fn.u_mul_e('nft', 'w', 'm'), fn.sum('m', 'nft'))
            block.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            n = block.dstdata['nft']
            ws = block.dstdata['ws'].unsqueeze(1).clamp(min=1)
            z = self.act(self.W(self.dropout(torch.cat([n / ws, h_dst], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z


class SAGENet(nn.Module):
    """
    NeighborSampler 클래스의 sample_blocks로 샘플링을 실행하면, blocks에 n-length의 리스트가 반환된다.
    blocks[0]는 Conv할 sub-graph에서 "2-depth neightbor nodes to 1-depth neighbor nodes"에 관한 정보를 가지고 있다.
    blocks[1] 또는 block[-1]은 "1-depth neighbor to traget nodes"에 관한 정보를 가지고 있다.
    이 계층적 구조에 맞게 Conv Layer를 n번 수행하는 반복 레이어 구조를 정의한 것이 SAGENet이다.
    """
    def __init__(self, hidden_dims, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims, hidden_dims))

    def forward(self, blocks, h):
        for layer, block in zip(self.convs, blocks):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst), block.edata['weights'])
        return h


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()
        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score
