import torch
import torch.nn as nn


# class TimeLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, batch_first=True):
#         super(TimeLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.W_all = nn.Linear(hidden_size, hidden_size * 4)
#         self.U_all = nn.Linear(input_size, hidden_size * 4)
#         self.W_d = nn.Linear(hidden_size, hidden_size)
#         assert batch_first == True

#     def forward(self, inputs, time_diff=None):
#         # inputs: [b, seq, embed]
#         # h: [b, hid]
#         # c: [b, hid]
#         b, seq, embed = inputs.size()
#         h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
#         c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
#         outputs = []
#         if time_diff is not None:
#             td_weight = 1 / torch.log(torch.exp(torch.ones(1).to(time_diff)) + time_diff) # Try time diff
#         for s in range(seq):
#             if time_diff is not None:
#                 # c_s1 = torch.tanh(self.W_d(c))
#                 # c_s2 = c_s1 * td_weight[:, s:s + 1].expand_as(c_s1)
#                 # c_l = c - c_s1
#                 # c_adj = c_l + c_s2
#                 c_adj = c * td_weight[:, s:s + 1].expand_as(c)
#             else:
#                 c_adj = c
#             outs = self.W_all(h) + self.U_all(inputs[:, s])
#             f, i, o, c_tmp = torch.chunk(outs, 4, 1)
#             f = torch.sigmoid(f)
#             i = torch.sigmoid(i)
#             o = torch.sigmoid(o)
#             c_tmp = torch.tanh(c_tmp) # ori: sigmoid
#             c = f * c_adj + i * c_tmp
#             h = o * torch.tanh(c)
#             outputs.append(h)
#         outputs = torch.stack(outputs, 1)
#         return outputs, (h.unsqueeze(0), c.unsqueeze(0))


class TimeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(TimeGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        assert batch_first == True

    def forward(self, inputs, time_diff=None, mask=None):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        outputs = []
        if time_diff is not None:
            td_weight = 1 / torch.log(torch.exp(torch.ones(1).to(time_diff)) + time_diff) # Try time diff
        for s in range(seq):
            if time_diff is not None:
                # c_s1 = torch.tanh(self.W_d(c))
                # c_s2 = c_s1 * td_weight[:, s:s + 1].expand_as(c_s1)
                # c_l = c - c_s1
                # c_adj = c_l + c_s2
                c_adj = c * td_weight[:, s:s + 1].expand_as(c)
            else:
                c_adj = c
            c_temp = self.gru_cell(inputs[:, s], c_adj)
            if mask is not None:
                c_temp[mask[:, s] == 1] = c[mask[:, s] == 1]
            c = c_temp
            h = c
            outputs.append(h)
        outputs = torch.stack(outputs, 1)
        return outputs, (h.unsqueeze(0), c.unsqueeze(0))
