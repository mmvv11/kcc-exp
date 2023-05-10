import os

import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import layers
from sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator


class PinSAGEModel(nn.Module):
    # TODO 1.
    """
    논문의 Projection Layer에 해당하는 부분을 slef.proj로 정의
    이 레이어는 노드의 input feature들을 projection 하는 레이어들의 모듈로 구성됌
    forward 함수는 positivenodes, negative nodes의 sub graph를 인자로 받는다.
    또한 이 노드들의 정보를 모두 포함하고 있는 block이라는 dgl 패키지의 자료구조를 입력 받는다.
    pos_graph, neg_graph는 일종의 포인터고, 실제 자료는 blocks

    그 결과 각각 임베딩 벡터를 만들어 낸다.
    Scorer 객체를 통하여 score를 계산하여 low-rank positive를 계산한다.

    """

    def __init__(self, full_graph, ntype, hidden_dims, n_layers):
        super().__init__()
        self.proj = layers.LinearProjector(full_graph, ntype, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_representation(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_representation(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset['train-graph']  # dgl graph
    user_ntype = dataset['user-type']  # 'user'
    item_ntype = dataset['item-type']  # 'movie'
    device = torch.device(args.device)

    # sampling
    batch_sampler = ItemToItemBatchSampler(g, user_ntype, item_ntype, args.batch_size)  # item에서 item으로 가는 랜덤 워크 샘플러
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)  # user의 이웃 노드를 핀세이지 알고리즘으로 샘플링
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)

    # model
    model = PinSAGEModel(g, item_ntype, args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_list = []

    # train in each batch
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.to(device) for block in blocks]
            loss = model(pos_graph, neg_graph, blocks).mean()
            loss_list.append(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print status
            if batch_id % 500 == 0:
                print("num_epochs:", epoch_id, "||", "batches_per_epoch:", batch_id, "||", "loss:", loss)

        # evaluate
        model.eval()
        with torch.no_grad():
            h_item_batches = []
            for blocks in dataloader_test:
                blocks = [block.to(device) for block in blocks]
                h_item_batches.append(model.get_representation(blocks))
            h_item = torch.cat(h_item_batches, 0)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict()
    }

    torch.save(checkpoint, "model.pt")

    return h_item
