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

        self.time_plus_weight = nn.Parameter(torch.zeros(1)) # Try time diff
        self.time_mul_weight = nn.Parameter(torch.ones(1)) # Try time diff

    def forward(self, q, k, v, time_diff, mask=None): # Try time diff
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # logging.info(torch.sum(time_diff <= 1), torch.sum(time_diff == 1), torch.sum(time_diff == 0), time_diff.numel())

        time_diff = time_diff / time_diff.mean() # Try time diff
        time_diff = time_diff + nn.functional.softplus(self.time_plus_weight) * torch.max(time_diff) # Try time diff
        time_diff_weight = 1 / torch.log(torch.exp(torch.ones(1).to(time_diff)) + time_diff) # Try time diff

        attn = attn + self.time_mul_weight * time_diff_weight # Try time diff

        # tdw_li = time_diff_weight.reshape(-1).tolist()
        # import matplotlib.pyplot as plt
        # # plt.hist(x = tdw_li, range=(0.99, 1.01), bins=100, color='steelblue', edgecolor='black')
        # plt.hist(x = tdw_li, bins=10, color='steelblue', edgecolor='black')
        # # # plt.hist(x = tdw_li, color='steelblue', edgecolor='black')
        # plt.show()

        # exit(0)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
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


    def forward(self, q, k, v, time_diff, mask=None): # Try time diff

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
        time_diff = time_diff.view(time_diff.shape[0], 1, time_diff.shape[1]).repeat(n_head, 1, 1) # Try time diff
        output, attn = self.attention(q, k, v, time_diff, mask=mask) # Try time diff

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)
        output = self.layer_norm(output)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)
        
        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)
        
        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]
        
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]
        
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk
        
        ## Map based Attention
        #output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]
        
        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn
    
def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        
        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        #self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        #torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()
        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len + 1, embedding_dim=expand_dim) # +1 for ts = 0
        self.seq_len = seq_len

    def forward(self, ts):
        # ts: [N, L]
        if torch.sum(torch.zeros_like(ts) == ts) == ts.numel():
            order = ts.long() + self.seq_len
        else:
            order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        # self.att_dim = feat_dim + edge_dim + time_dim
        self.att_dim = feat_dim + edge_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, _, mask): # Try time diff
        # seq [B, N, D]
        # mask [B, N]
        # seq_x = torch.cat([seq, seq_e, seq_t], dim=2)
        seq_x = torch.cat([seq, seq_e], dim=2)

        _, (hn, _) = self.lstm(seq_x)

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

    def forward(self, src, src_t, seq, seq_t, seq_e, _, mask): # Try time diff
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
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        #self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        #self.act = torch.nn.ReLU()

        assert(self.model_dim % n_head == 0)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            logging.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            logging.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')
        
    def forward(self, src, src_t, seq, seq_t, seq_e, time_diff, mask): # Try time diff
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

        src_ext = torch.unsqueeze(src, dim=1) # src [B, 1, D]
        # src_e_ph = torch.zeros_like(src_ext)
        src_e_ph = torch.zeros(src_ext.shape[0], src_ext.shape[1], self.edge_dim).to(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2) # [B, 1, D + De + Dt]
        k = torch.cat([seq, seq_e, seq_t], dim=2) # [B, N, D + De + Dt]

        mask = torch.unsqueeze(mask, dim=2) # mask [B, N, 1]
        mask = mask.permute([0, 2, 1]) #mask [B, 1, N]

        # # target-attention
        # output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, 1, D + Dt], attn: [B, 1, N]
        output, attn = self.multi_head_target(q=q, k=k, v=k, time_diff=time_diff, mask=mask) # Try time diff
        output = output.squeeze()
        attn = attn.squeeze()

        output = self.merger(output, src)
        return output, attn


class MixModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, 
                 attn_mode='prod', n_head=2, drop_out=0.1):
        super(MixModel, self).__init__()
        self.attn_model = AttnModel(feat_dim, edge_dim, time_dim, attn_mode, n_head, drop_out)
        self.lstm_model = LSTMPool(feat_dim, edge_dim, time_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, time_diff, mask): # Try time diff
        attn_result, _ = self.attn_model(src, src_t, seq, seq_t, seq_e, time_diff, mask) # Try time diff
        lstm_result, _ = self.lstm_model(src, src_t, seq, seq_t, seq_e, time_diff, mask) # Try time diff

        output = (attn_result + lstm_result) / 2 # TODO: better merge
        # output = torch.max(torch.stack((attn_result, lstm_result), dim=1), dim=1)[0] # use better merge
        return output, None


class TGCN(torch.nn.Module):
    def __init__(self, ngh_finder, feat_dim, edge_dim, time_dim, n_node, n_edge, device='cpu', num_layers=3, num_workers=0,
                 pos_encoder='time', agg_method='attn', attn_mode='prod', n_head=4, drop_out=0.1, seq_len=None):
        super(TGCN, self).__init__()
        self.workers_alive = False

        self.ngh_finder = ngh_finder
        self.feat_dim = feat_dim # feature_dim
        self.edge_dim = edge_dim # edge_dim
        self.time_dim = time_dim # time_dim
        self.device = device
        self.num_layers = num_layers

        if num_workers is None:
            self.num_workers = None
        else:
            self.num_workers = num_workers if num_workers != 0 else cpu_count() // 2

        self.node_embed = torch.nn.Embedding(num_embeddings=n_node, embedding_dim=self.feat_dim)
        # self.edge_embed = torch.nn.Embedding(num_embeddings=n_edge, embedding_dim=self.feat_dim)
        self.edge_embed = torch.nn.Embedding(num_embeddings=n_edge, embedding_dim=self.edge_dim)

        # Choose position encoder
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

        # Choose aggregate method
        if agg_method == 'attn':
            logging.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.edge_dim,
                                                                  self.time_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) for _ in range(num_layers)])
        elif agg_method == 'lstm':
            logging.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.edge_dim,
                                                                 self.time_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            logging.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.edge_dim) for _ in range(num_layers)])
        elif agg_method == 'mix':
            logging.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([MixModel(self.feat_dim,
                                                                  self.edge_dim,
                                                                  self.time_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) for _ in range(num_layers)])
        else:
            raise ValueError('invalid agg_method value, use attn or lstm')

        # self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)

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
        pos_scores = torch.sum(src_embed * tgt_embed, dim=1)
        neg_scores = torch.sum(src_embed * neg_embed, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        # loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores))) # same as softplus
        # reg_loss = (1/2) * reg_loss / float(len(users))
        return loss

    def get_top_n(self, src_nodes, candidate_nodes, cut_times, num_neighbors=20, topk=20):
        src_embed = self.tem_conv(src_nodes, cut_times, self.num_layers, 0, num_neighbors)
        batch_ratings = []
        for cad, cut, src in zip(candidate_nodes, cut_times, src_embed):
            cut = np.tile(cut, (cad.shape[0]))
            candidate_embed = self.tem_conv(cad, cut, self.num_layers, 0, num_neighbors)
            ratings = torch.matmul(candidate_embed, src) # shape=(101,)
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
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   curr_distance=curr_distance + 1,
                                                   num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            if self.num_layers >= 3 and curr_distance + 1 == self.num_layers:
                src_ngh_feat = src_ngh_feat[:, 10:, :]
                src_ngh_node_batch_th = src_ngh_node_batch_th[:, 10:]
                src_ngh_t_batch_th = src_ngh_t_batch_th[:, 10:]
                src_ngh_eidx_batch = src_ngh_eidx_batch[:, 10:]

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   src_ngh_t_batch_th, # Try time diff
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
