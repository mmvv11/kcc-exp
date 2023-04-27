import dgl
import torch
from torch.utils.data import IterableDataset


class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        # TODO 2.
        """
        핀세이지 샘플러 클래스로 샘플링을 하려면, 샘플링의 대상이 되는 seed id들을 우선 입력해야한다.
        그래서 샘플링에 필요한 Positive node, Negative node에 해당하는 seed ids를 우선 정의한 뒤, 샘플링을 진행한다.
        Positive 노드는 Target 노드로부터 Random Walk를 수행하여 n-depth 시뮬레이션 내에 포함되는 노드를 의미하고,
        Negative 노드는 그래프 연결관계와 상관 없이 완전히 랜덤으로 샘플링된 노드를 의미한다.
        데이터셋이 커지면 커질수록 Positive-Negative를 구분하는 것이 확률 적으로 정확해진다.
        seed ids를 출력하는 코드는 아래와 같다.
        heads가 시작 노드
        tails가 positive nodes
        neg_tails는 negative nodes를 의미한다.

        """
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            result = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])
            tails = result[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(g=frontier, dst_nodes=seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]


def assign_features_to_blocks(blocks, g, ntype):
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)


class NeighborSampler(object):
    # TODO 1.
    """
    GraphSAGE 알고리즘은 Conv 연산을 수행할 수 있는 형태로, 인접 노드를 샘플링 해야한다.
    핀세이지는 PPR에 기반한 Random walks로 추출한다.
    즉, 하나의 노드를 샘플링 할 때는 Page rank edge score를 기준으로 링크가 강한 연결관계를 위주로 랜덤 샘플링을 진행한다.
    이는 dgl로 손쉽게 사용 가능.
    아래 코드에서 self.samplers 부분이 바로 핀세이지 샘플링 객체 선언 부분임.
    """
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                                        random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(frontier, eids)
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block) 
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # TODO 3.
        """
        모델을 학습하는 과정에서는 positive nodes와 negative nodes를 각각 모델에 적용해 low-rank-positives 랭킹 학습을 수행하기 때문에,
        샘플링과 동시에 학습 과정에서 분리할 positive sub-graph, negative sub-graph도 각각 정의 한다.

        :param heads: 시작 노드
        :param tails: positive nodes
        :param neg_tails: negative nodeds
        :return:
        """
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs(
            [pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]
        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks



class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return blocks
