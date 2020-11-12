import os
import logging
import multiprocessing
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

from global_flag import get_flag
from order_encoder import TimeEncode, PosEncode, EmptyEncode
from t_lstm import TimeLSTM


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        #x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.time_plus_weight = nn.Parameter(torch.zeros(1))
        self.time_mul_weight = nn.Parameter(torch.ones(1))

    def forward(self, q, k, v, time_diff=None, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if torch.sum(mask) != 0:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn) # [n * b, l_q, l_k]

        # if time_diff is not None:
        if False:
            time_diff = time_diff / time_diff.mean()
            time_diff = time_diff + nn.functional.softplus(self.time_plus_weight) * torch.max(time_diff)
            time_diff_weight = 1 / torch.log(torch.exp(torch.ones(1).to(time_diff)) + time_diff)
            if torch.sum(mask) != 0:
                time_diff_weight = time_diff_weight.masked_fill(mask, -1e10)
            time_diff_weight = self.softmax(time_diff_weight)
            attn = (attn + self.time_mul_weight * time_diff_weight) / (1 + self.time_mul_weight)

        attn = self.dropout(attn) # [n * b, l_v, d]
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, time_diff=None, mask=None, use_res=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        if time_diff is not None:
            time_diff = time_diff.view(time_diff.shape[0], 1, time_diff.shape[1]).repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, time_diff, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if use_res:
            output = output + residual
        output = self.layer_norm(output)

        return output, attn


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim, data_set='newAmazon'):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        # self.att_dim = feat_dim + edge_dim + time_dim
        self.att_dim = feat_dim + edge_dim

        self.act = torch.nn.ReLU()

        # self.lstm = torch.nn.LSTM(input_size=self.att_dim,
        #                           hidden_size=self.feat_dim,
        #                           num_layers=1,
        #                           batch_first=True)
        self.lstm = TimeLSTM(input_size=self.att_dim, hidden_size=self.feat_dim, batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)
        self.data_set = data_set

    def forward(self, src, src_t, seq, seq_t, seq_e, time_diff, mask):
        # seq [B, N, D]
        # mask [B, N]
        # seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
        seq_x = torch.cat([seq, seq_e], dim=2)

        if time_diff is not None:
            time_span = time_diff.clone()
            for i in range(1, time_diff.shape[1]):
                if self.data_set == 'newAmazon':
                    new_diff = (time_diff[:, time_diff.shape[1] - i - 1] - time_diff[:, time_diff.shape[1] - i]) / 5000000 # for newAmazon
                elif self.data_set == 'goodreads_large':
                    new_diff = (time_diff[:, time_diff.shape[1] - i - 1] - time_diff[:, time_diff.shape[1] - i]) / 1000000 # for goodreads_large
                else:
                    new_diff = (time_diff[:, time_diff.shape[1] - i - 1] - time_diff[:, time_diff.shape[1] - i]) / 1000000 # for goodreads_large
                    # assert False, 'False data_set'
                # new_diff_nz = new_diff[new_diff != 0]
                # new_diff_0 = new_diff_nz[new_diff_nz.int() == 0]
                # new_diff_1 = new_diff[new_diff.int() == 1]
                # new_diff_2 = new_diff[new_diff.int() == 2]
                # new_diff_3 = new_diff[new_diff.int() == 3]
                # new_diff_10 = new_diff[new_diff.int() >= 10]
                # print(new_diff_nz.int().tolist())
                # print(new_diff.numel(), new_diff_nz.numel(), new_diff_0.numel(), new_diff_1.numel(), new_diff_2.numel(), new_diff_3.numel(), new_diff_10.numel())
                # print(new_diff_nz.min(), new_diff_nz.max(), new_diff_nz.mean(), new_diff_nz.shape)
                # print('--------------------------------------------------')
                # import time
                # time.sleep(1)
                time_span[:, time_diff.shape[1] - i] = new_diff
            time_span[:, 0] = torch.zeros_like(time_diff[:, 0])
            assert torch.sum(time_span >= 0) == time_span.numel()
            _, (hn, _) = self.lstm(seq_x, time_span, mask) # for TimeLSTM
        else:
            _, (hn, _) = self.lstm(seq_x) # for torch.nn.LSTM

        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, _, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 n_head=2, drop_out=0.1, sa_layers=0):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        self.transformer_dim = feat_dim + time_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()

        assert(self.model_dim % n_head == 0)

        self.sa_layers = sa_layers # use self-attention
        if sa_layers != 0:
            self.transformer_modules = nn.ModuleList([
                MultiHeadAttention(n_head, self.transformer_dim, self.transformer_dim // n_head,
                                self.transformer_dim // n_head, dropout=drop_out)
                for _ in range(sa_layers)])

        self.multi_head_target = MultiHeadAttention(n_head,
                                            d_model=self.model_dim,
                                            d_k=self.model_dim // n_head,
                                            d_v=self.model_dim // n_head,
                                            dropout=drop_out)

    def forward(self, src, src_t, seq, seq_t, seq_e, time_diff, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        # src_e_ph = torch.zeros_like(src_ext)
        src_e_ph = torch.zeros(src_ext.shape[0], src_ext.shape[1], self.edge_dim).to(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt]

        if self.sa_layers == 0:
            k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N, D + De + Dt]
        else:
            t_k_list = []
            t_k = torch.cat([seq, seq_t], dim=2) # [B, N, D + Dt]
            for i in range(self.sa_layers):
                t_k, _ = self.transformer_modules[i](q=t_k, k=t_k, v=t_k, time_diff=None, mask=mask, use_res=True)
                t_k_list.append(t_k)
            t_k = torch.stack(t_k_list, dim=0).mean(dim=0)
            k = torch.cat((t_k[:, :, :seq.shape[2]], seq_e, t_k[:, :, seq.shape[2]:]), dim=-1)

        # # target-attention
        # output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output, attn = self.multi_head_target(q=q, k=k, v=k, time_diff=time_diff, mask=mask)
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class MixModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 n_head=2, drop_out=0.1, sa_layers=0, data_set='newAmazon'):
        super(MixModel, self).__init__()
        self.attn_model = AttnModel(feat_dim, edge_dim, time_dim, n_head, drop_out, sa_layers)
        self.lstm_model = LSTMPool(feat_dim, edge_dim, time_dim, data_set=data_set)

        # self.weight_attn = nn.Parameter(torch.zeros(1) + 0.5)
        # self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, time_diff, mask):
        attn_result, _ = self.attn_model(src, src_t, seq, seq_t, seq_e, time_diff, mask)
        lstm_result, _ = self.lstm_model(src, src_t, seq, seq_t, seq_e, time_diff, mask)

        output = (attn_result + lstm_result) / 2 # Mean
        # output = self.weight_attn * attn_result + (1 - self.weight_attn) * lstm_result # Weighted sum
        # output = torch.max(torch.stack((attn_result, lstm_result), dim=1), dim=1)[0] # Max pool
        # output = self.merger(attn_result, lstm_result) # Concat & fc

        # if get_flag():
        #     print('Mix weight: ', self.weight_attn.data.item())

        return output, None


class PruneModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim):
        super(PruneModel, self).__init__()
        self.model_dim = feat_dim
        self.attn_module = MultiHeadAttention(1, self.model_dim, self.model_dim, self.model_dim)
        self.src = None

    def set_src(self, src, n_ngh):
        self.src = src.unsqueeze(1).repeat(1, n_ngh, 1).flatten(0, 1)

    def forward(self, seq, mask):
        assert self.src is not None
        mask_temp = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask_temp = mask_temp.permute([0, 2, 1]) #mask [B, 1, N]
        src_ext = torch.unsqueeze(self.src, dim=1)
        _, attn = self.attn_module(q=src_ext, k=seq, v=seq, time_diff=None, mask=mask_temp, use_res=False)
        # print(attn.shape, mask.shape, mask_temp.shape) # torch.Size([30720, 1, 40]) torch.Size([30720, 40]) torch.Size([30720, 1, 40])
        attn = attn.squeeze()
        mask[attn < (attn.mean(dim=1).unsqueeze(dim=1) / 5)] = 1
        return mask


class TGCN(torch.nn.Module):
    def __init__(self, ngh_finder, feat_dim, edge_dim, time_dim, n_node, n_edge, device='cpu', num_layers=3, use_td=False, target_mode='prod', maigin=10, prune=False,
                 num_workers=0, pos_encoder='time', agg_method='attn', n_head=4, drop_out=0.1, seq_len=None, sa_layers=0, data_set='newAmazon'):
        super(TGCN, self).__init__()
        self.workers_alive = False

        self.ngh_finder = ngh_finder
        self.feat_dim = feat_dim # feature_dim
        self.edge_dim = edge_dim # edge_dim
        self.time_dim = time_dim # time_dim
        self.device = device
        self.num_layers = num_layers
        self.use_td = use_td
        self.target_mode = target_mode
        self.maigin = maigin
        self.prune = prune

        if num_workers is None:
            self.num_workers = None
        else:
            self.num_workers = num_workers if num_workers != 0 else cpu_count() // 2

        self.node_embed = torch.nn.Embedding(num_embeddings=n_node, embedding_dim=self.feat_dim)
        # self.edge_embed = torch.nn.Embedding(num_embeddings=n_edge, embedding_dim=self.feat_dim)
        self.edge_embed = torch.nn.Embedding(num_embeddings=n_edge, embedding_dim=self.edge_dim)

        # Choose position encoder
        if self.time_dim != 0:
            if pos_encoder == 'time':
                logging.info('Using time encoding')
                self.time_encoder = TimeEncode(expand_dim=self.time_dim)
            elif pos_encoder == 'pos':
                assert(seq_len is not None)
                logging.info('Using positional encoding')
                self.time_encoder = PosEncode(expand_dim=self.time_dim, seq_len=seq_len)
            elif pos_encoder == 'empty':
                logging.info('Using empty encoding')
                self.time_encoder = EmptyEncode(expand_dim=self.time_dim)
            else:
                raise ValueError('invalid pos_encoder option!')
        else:
            self.time_encoder = EmptyEncode(expand_dim=self.time_dim)

        # Choose aggregate method
        if agg_method == 'attn':
            logging.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.edge_dim,
                                                                  self.time_dim,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out,
                                                                  sa_layers=sa_layers) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            logging.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.edge_dim,
                                                                 self.time_dim,
                                                                 data_set=data_set) for _ in range(num_layers)])
        elif agg_method == 'mean':
            logging.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.edge_dim) for _ in range(num_layers)])
        elif agg_method == 'mix':
            logging.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([MixModel(self.feat_dim,
                                                                  self.edge_dim,
                                                                  self.time_dim,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out,
                                                                  sa_layers=sa_layers,
                                                                  data_set=data_set) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value, use attn or lstm')

        # self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        self.prune_model = PruneModel(self.feat_dim)

    def init_workers(self):
        if self.num_workers is not None:
            assert not self.workers_alive
            self.index_queues = []
            self.workers = []
            self.data_queue = multiprocessing.Queue()
            for i in range(self.num_workers):
                index_queue = multiprocessing.Queue()
                w = multiprocessing.Process(
                    target=_workers,
                    args=(self.ngh_finder, index_queue, self.data_queue))
                w.daemon = True
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)
            self.workers_alive = True

    def _organize_received_data(self):
        assert self.workers_alive
        result_dict = {}
        while True:
            data = self.data_queue.get(timeout=100)
            result_dict[data[0]] = data[1:]
            if len(result_dict) == self.num_workers:
                break
        return [result_dict[i] for i in range(self.num_workers)]


    def del_workers(self):
        if self.num_workers is not None:
            assert self.workers_alive
            for i in range(self.num_workers):
                self.index_queues[i].put((i, None))
            received_data = self._organize_received_data()
            for i in range(self.num_workers):
                assert received_data[i][0] is None
                self.workers[i].join()
                self.index_queues[i].close()
            self.data_queue.close()

            self.index_queues = None
            self.workers = None
            self.data_queue = None
            self.workers_alive = False

    def bpr_loss(self, src_nodes, tgt_nodes, neg_nodes, cut_times, num_neighbors=20):
        src_embed = self.tem_conv(src_nodes, cut_times, self.num_layers, 0, num_neighbors)
        tgt_embed = self.tem_conv(tgt_nodes, cut_times, self.num_layers, 0, num_neighbors)
        neg_embed = self.tem_conv(neg_nodes, cut_times, self.num_layers, 0, num_neighbors)
        if self.target_mode == 'prod':
            pos_scores = torch.sum(src_embed * tgt_embed, dim=1)
            neg_scores = torch.sum(src_embed * neg_embed, dim=1)
            loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
            # loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores))) # same as softplus
        elif self.target_mode == 'dist':
            pos_dist = F.pairwise_distance(src_embed, tgt_embed, 2)
            neg_dist = F.pairwise_distance(src_embed, neg_embed, 2)
            loss = torch.sum(F.relu(pos_dist - neg_dist + self.maigin))
        else:
            assert False, 'False target_mode'
        return loss

    def get_top_n(self, src_nodes, candidate_nodes, cut_times, num_neighbors=20, topk=20):
        src_embed = self.tem_conv(src_nodes, cut_times, self.num_layers, 0, num_neighbors)
        batch_ratings = []
        for cad, cut, src in zip(candidate_nodes, cut_times, src_embed):
            cut = np.tile(cut, (cad.shape[0]))
            candidate_embed = self.tem_conv(cad, cut, self.num_layers, 0, num_neighbors)
            if self.target_mode == 'prod':
                ratings = torch.matmul(candidate_embed, src) # shape=(101,)
            elif self.target_mode == 'dist':
                distance = F.pairwise_distance(candidate_embed, src, 2)
                ratings = -distance
            else:
                assert False, 'False target_mode'
            batch_ratings.append(ratings)
        batch_ratings = torch.stack(batch_ratings) # shape=(batch_size, candidate_size)
        if topk <= 0:
            topk = batch_ratings.shape[1]
        rate_k, index_k = torch.topk(batch_ratings, k=topk) # index_k.shape = (batch_size, TOPK), dtype=torch.int
        batch_topk_ids = torch.gather(torch.Tensor(candidate_nodes).long().to(self.device), 1, index_k)
        return batch_topk_ids

    def tem_conv(self, src_nodes, cut_times, curr_layers, curr_distance, num_neighbors=20):
        assert(curr_layers >= 0)
        assert src_nodes.ndim == 1
        assert cut_times.ndim == 1

        batch_size = len(src_nodes)

        src_node_batch_th = torch.from_numpy(src_nodes).long().to(self.device)
        cut_time_l_th = torch.from_numpy(cut_times).float().to(self.device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_embed(src_node_batch_th)

        if curr_layers == 0:
            if curr_distance == 0:
                self.prune_model.set_src(src_node_feat, num_neighbors)
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_nodes,
                                               cut_times,
                                               curr_layers=curr_layers - 1,
                                               curr_distance=curr_distance,
                                               num_neighbors=num_neighbors)


            if self.num_workers is None:
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(src_nodes,
                                                                                                                cut_times,
                                                                                                                num_neighbors)
            else:
                src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.parallel_ngh_find(src_nodes, cut_times, num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(self.device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(self.device)

            src_ngh_t_batch_delta = cut_times[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(self.device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            # print(src_ngh_node_batch.shape, src_ngh_node_batch_flat.shape)
            # exit(0)

            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)
            # src_ngh_t_batch_flat = (cut_times[:, np.newaxis] - np.zeros_like(src_ngh_t_batch)).flatten()
            # src_ngh_t_batch_flat = (cut_times[:, np.newaxis] - (src_ngh_t_batch_delta / 2)).flatten()

            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   curr_distance=curr_distance + 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            mask = src_ngh_node_batch_th == 0

            # if num_neighbors >= 30 and curr_distance + 1 == self.num_layers:
            if self.prune and curr_distance + 1 == self.num_layers:
                mask = self.prune_model(src_ngh_feat, mask)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_embed(src_ngh_eidx_batch)

            # attention aggregation
            attn_m = self.attn_model_list[curr_layers - 1]

            if self.use_td:
                time_diff = src_ngh_t_batch_th
            else:
                time_diff = None

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   time_diff,
                                   mask)
            return local

    def parallel_ngh_find(self, src_nodes, cut_times, num_neighbors):

        def partition_array(src_nodes, cut_times, n_workers):
            assert len(src_nodes) == len(cut_times)
            batch_size = (len(src_nodes) - 1) // n_workers + 1
            part_list_node = []
            part_list_time = []
            count = 0
            for i in range(n_workers):
                if i == 0:
                    part_list_node.append(src_nodes[: batch_size])
                    part_list_time.append(cut_times[: batch_size])
                elif i == n_workers - 1:
                    part_list_node.append(src_nodes[i * batch_size:])
                    part_list_time.append(cut_times[i * batch_size:])
                else:
                    part_list_node.append(src_nodes[i * batch_size: (i+1) * batch_size])
                    part_list_time.append(cut_times[i * batch_size: (i+1) * batch_size])
            return part_list_node, part_list_time

        assert self.workers_alive
        part_list_node, part_list_time = partition_array(src_nodes, cut_times, self.num_workers)
        for i in range(self.num_workers):
            self.index_queues[i].put((i, part_list_node[i], part_list_time[i], num_neighbors))
        received_data = self._organize_received_data()

        out_neighbors = np.concatenate([x[0] for x in received_data])
        baout_edges = np.concatenate([x[1] for x in received_data])
        out_timestamps = np.concatenate([x[2] for x in received_data])
        return out_neighbors, baout_edges, out_timestamps


class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


def _workers(finder, index_queue, data_queue):
    watchdog = ManagerWatchdog()
    while watchdog.is_alive():
        # try:
        #     index_in = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        # except:
        #     assert False # continue
        index_in = index_queue.get(timeout=100)

        if index_in[1] is None:
            job_id = index_in[0]
            break
        job_id, part_list_node, part_list_time, num_neighbors = index_in
        n, e, t = finder.get_temporal_neighbor(part_list_node, part_list_time, num_neighbors)
        data_queue.put((job_id, n, e, t))
    data_queue.put((job_id, None))
