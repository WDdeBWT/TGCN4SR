import math

import numpy as np
from collections import defaultdict

import torch

data_path = 'tisasrec_data/'

# class TrainDataset(torch.utils.data.Dataset):

#     def __init__(self, adj_list, n_user, n_item, min_train_seq):
#         import random
#         self.adj_list = adj_list
#         self.n_user = n_user
#         self.n_item = n_item
#         self.min_train_seq = min_train_seq
#         self.test_temp(adj_list, n_user, n_item, min_train_seq)

#     def shuffle_temp(self):
#         self.test_temp(self.adj_list, self.n_user, self.n_item, self.min_train_seq)

#     def test_temp(self, adj_list, n_user, n_item, min_train_seq):
#         self.instance_user = []
#         self.instance_item = []
#         self.instance_time = []
#         self.user_map_only_item = defaultdict(list)

#         u_list = list(adj_list.keys())
#         random.shuffle(u_list)

#         for u in u_list:
#             if u >= n_user:
#                 continue
#             assert len(adj_list[u]) > min_train_seq
#             sorted_tuple = sorted(adj_list[u], key=lambda x: x[2])
#             # for x in sorted_tuple[2:]: # TODO: Try not use [2:]
#             #     self.instance_user.append(u)
#             #     self.instance_item.append(x[0])
#             #     self.instance_time.append(x[2])
#             # for i in range(2, len(sorted_tuple)):
#             for i in range(min_train_seq - 1, len(sorted_tuple)):
#                 self.instance_user.append(u)
#                 self.instance_item.append(sorted_tuple[i][0])
#                 self.instance_time.append(sorted_tuple[i - 1][2] + 1)
#                 # self.instance_time.append(sorted_tuple[i][2])
#             self.user_map_only_item[u] = [x[0] for x in sorted_tuple]
#         assert len(self.instance_user) == len(self.instance_item)
#         assert len(self.instance_user) == len(self.instance_time)
#         self.n_user = n_user
#         self.n_item = n_item

#     def __len__(self):
#         return len(self.instance_user)

#     def __getitem__(self, index):
#         user_id = self.instance_user[index]
#         pos_id = self.instance_item[index]
#         time_stamp = self.instance_time[index]
#         while True:
#             neg_id = np.random.randint(self.n_user, self.n_user + self.n_item)
#             if neg_id in self.user_map_only_item[user_id]:
#                 continue
#             else:
#                 break
#         return user_id, pos_id, neg_id, time_stamp


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, adj_list, n_user, n_item, min_train_seq):
        self.instance_user = []
        self.instance_item = []
        self.instance_time = []
        self.user_map_only_item = defaultdict(list)
        for u in adj_list:
            if u >= n_user:
                continue
            assert len(adj_list[u]) > min_train_seq
            sorted_tuple = sorted(adj_list[u], key=lambda x: x[2])
            # for x in sorted_tuple[2:]: # TODO: Try not use [2:]
            #     self.instance_user.append(u)
            #     self.instance_item.append(x[0])
            #     self.instance_time.append(x[2])
            # for i in range(2, len(sorted_tuple)):
            for i in range(min_train_seq - 1, len(sorted_tuple)):
                self.instance_user.append(u)
                self.instance_item.append(sorted_tuple[i][0])
                # self.instance_time.append(sorted_tuple[i - 1][2] + 1)
                self.instance_time.append(sorted_tuple[i][2])
            self.user_map_only_item[u] = [x[0] for x in sorted_tuple]
        assert len(self.instance_user) == len(self.instance_item)
        assert len(self.instance_user) == len(self.instance_time)
        self.n_user = n_user
        self.n_item = n_item

    def __len__(self):
        return len(self.instance_user)

    def __getitem__(self, index):
        user_id = self.instance_user[index]
        pos_id = self.instance_item[index]
        time_stamp = self.instance_time[index]
        while True:
            neg_id = np.random.randint(self.n_user, self.n_user + self.n_item)
            if neg_id in self.user_map_only_item[user_id]:
                continue
            else:
                break
        return user_id, pos_id, neg_id, time_stamp


class ValidDataset(torch.utils.data.Dataset):

    def __init__(self, adj_list, n_user, n_item):
        self.instance_user = []
        self.instance_item = []
        self.instance_time = []
        self.user_map_only_item = defaultdict(list)
        for u in adj_list:
            if u >= n_user:
                continue
            sorted_tuple = sorted(adj_list[u], key=lambda x: x[2])
            self.instance_user.append(u)
            self.instance_item.append(sorted_tuple[-1][0])
            # self.instance_time.append(sorted_tuple[-2][2] + 1)
            self.instance_time.append(sorted_tuple[-1][2])
            self.user_map_only_item[u] = [x[0] for x in sorted_tuple]
        assert len(self.instance_user) == len(self.instance_item)
        assert len(self.instance_user) == len(self.instance_time)
        self.n_user = n_user
        self.n_item = n_item

    def __len__(self):
        return len(self.instance_user)

    def __getitem__(self, index):
        user_id = self.instance_user[index]
        pos_id = self.instance_item[index]
        time_stamp = self.instance_time[index]
        while True:
            neg_id = np.random.randint(self.n_user, self.n_user + self.n_item)
            if neg_id in self.user_map_only_item[user_id]:
                continue
            else:
                break
        return user_id, pos_id, neg_id, time_stamp


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, adj_list, test_candidate, n_user, n_item=None):
        # TODO: delete n_item
        self.test_instance_user = []
        self.test_instance_target = []
        self.test_instance_candidate = []
        self.test_instance_time = []
        self.user_map_only_item = defaultdict(list)
        for u in adj_list:
            if u >= n_user:
                continue
            sorted_tuple = sorted(adj_list[u], key=lambda x: x[2])
            assert u in test_candidate
            x = sorted_tuple[-1]
            self.test_instance_user.append(u)
            self.test_instance_target.append(sorted_tuple[-1][0])
            self.test_instance_candidate.append(test_candidate[u])
            # self.test_instance_time.append(sorted_tuple[-2][2] + 1)
            self.test_instance_time.append(sorted_tuple[-1][2])
            self.user_map_only_item[u] = [x[0] for x in sorted_tuple]
        assert len(self.test_instance_user) == len(self.test_instance_target)
        assert len(self.test_instance_user) == len(self.test_instance_candidate)
        assert len(self.test_instance_user) == len(self.test_instance_time)
        self.n_user = n_user
        self.n_item = n_item

    def __len__(self):
        return len(self.test_instance_user)

    # def __getitem__(self, index):
    #     user_id = self.test_instance_user[index]
    #     target_id = self.test_instance_target[index]
    #     candidate_ids = torch.Tensor(self.test_instance_candidate[index]).long()
    #     time_stamp = self.test_instance_time[index]
    #     return user_id, target_id, candidate_ids, time_stamp

    def __getitem__(self, index):
        # sample version
        user_id = self.test_instance_user[index]
        target_id = self.test_instance_target[index]
        time_stamp = self.test_instance_time[index]
        candidate_ids = []
        while len(candidate_ids) < 100:
            neg_id = np.random.randint(self.n_user, self.n_user + self.n_item)
            if neg_id in self.user_map_only_item[user_id]:
                continue
            else:
                candidate_ids.append(int(neg_id))
        candidate_ids.append(int(target_id))
        candidate_ids = torch.Tensor(candidate_ids).long()
        return user_id, target_id, candidate_ids, time_stamp


def data_partition_amz(dataset_name='newAmazon'):
    n_user = 0
    n_item = 0
    adj_list_original = defaultdict(list)
    adj_list_train = defaultdict(list) # train data for valid
    adj_list_tandv = defaultdict(list) # full = train+valid, as the train data for test
    adj_list_tavat = defaultdict(list) # full = train+valid+test, as the full adj
    # user_map_train = {}
    # user_map_valid = {}
    # user_map_test = {}
    test_candidate = {}

    # assume user/item index starting from 1
    path_to_data = data_path + dataset_name + '/' + dataset_name + '_all.txt'

    f = open(path_to_data, 'r')
    for line in f:
        u, i, t, d = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        t = int(t)

        n_user = max(u, n_user)
        n_item = max(i, n_item)
        adj_list_original[u].append((i, t))
    f.close()

    min_nfeedback = 10
    total_feedback = 0
    for user in adj_list_original:
        adj_list_original[user].sort(key=lambda x: x[1])
        nfeedback = len(adj_list_original[user])
        total_feedback += nfeedback
        assert nfeedback >= 5
        if nfeedback < min_nfeedback:
            min_nfeedback = nfeedback
        # user_map_train[user] = [(x[0], x[1]) for x in adj_list_original[user][:-2]]
        # user_map_valid[user] = [(adj_list_original[user][-2][0], adj_list_original[user][-2][1])]
        # user_map_test[user] = [(adj_list_original[user][-1][0], adj_list_original[user][-1][1])]

        test_candidate[user] = [adj_list_original[user][-1][0]]
    print('min_nfeedback:', min_nfeedback, '- total_feedback:', total_feedback)

    skip = 0
    neg_f = data_path + dataset_name + '/' + dataset_name + '_test_neg.txt'
    f = open(neg_f, 'r')
    for line in f:
        skip += 1
        if skip == 1:
            continue
        user_id, item_id = line.rstrip().split('\t')
        u = int(user_id)
        i = int(item_id)
        n_user = max(u, n_user)
        n_item = max(i, n_item)

        test_candidate[u].append(i)
    f.close()

    n_user = n_user + 1
    n_item = n_item + 1

    for user in adj_list_original:
        # adj_list_original[user] = [(x[0] + n_user, 0, x[1]) for x in adj_list_original[user]]
        # # user_map_train[user] = [(x[0] + n_user, x[1]) for x in user_map_train[user]]
        # # user_map_valid[user] = [(x[0] + n_user, x[1]) for x in user_map_valid[user]]
        # user_map_test[user] = [(x[0] + n_user, x[1]) for x in user_map_test[user]]
        test_candidate[user] = [x + n_user for x in test_candidate[user]]

        adj_list_train[user] = [(x[0] + n_user, 0, x[1]) for x in adj_list_original[user][:-2]]
        for x in adj_list_train[user]:
            adj_list_train[x[0]].append((user, 1, x[2]))
        adj_list_tandv[user] = [(x[0] + n_user, 0, x[1]) for x in adj_list_original[user][:-1]]
        for x in adj_list_tandv[user]:
            adj_list_tandv[x[0]].append((user, 1, x[2]))
        adj_list_tavat[user] = [(x[0] + n_user, 0, x[1]) for x in adj_list_original[user]]
        for x in adj_list_tavat[user]:
            adj_list_tavat[x[0]].append((user, 1, x[2]))

    return adj_list_train, adj_list_tandv, adj_list_tavat, test_candidate, n_user, n_item


def statistic_dataset(adj_list):
    total_min = float('inf')
    total_max = -float('inf')
    sequence_time_span = []
    sequence_time_interval = []
    for user in adj_list:
        assert len(adj_list[user]) >= 1
        min_ts = float('inf')
        max_ts = -float('inf')
        temp_list = []
        for x in adj_list[user]:
            if x[2] < min_ts:
                min_ts = x[2]
            if x[2] > max_ts:
                max_ts = x[2]
            temp_list.append(x[2])
        if min_ts < total_min:
            total_min = min_ts
        if max_ts > total_max:
            total_max = max_ts
        temp_list.sort()
        for i in range(len(temp_list) - 1):
            sequence_time_interval.append(temp_list[i + 1] - temp_list[i])
        sequence_time_span.append(max_ts - min_ts)
    sequence_time_span = np.array(sequence_time_span) / 31536000
    # sequence_time_interval = np.array(sequence_time_interval) / 31536000
    sequence_time_interval = np.array(sequence_time_interval)
    sequence_time_interval = sequence_time_interval / sequence_time_interval.mean()
    print('----- whole dataset -----')
    print('min stamp:', total_min, ', max stamp:', total_max, ', time span(year):', ((total_max - total_min) / 31536000))
    print('----- each user\'s time span -----')
    print('mean & var', sequence_time_span.mean(), sequence_time_span.var())
    print('----- each interaction\'s time interval -----')
    print('mean & var & max', sequence_time_interval.mean(), sequence_time_interval.var(), sequence_time_interval.max())


if __name__ == "__main__":
    # amazon_cds_vinyl, amazon_movies_tv, amazon_beauty, amazon_game, steam
    dataset = 'steam'
    print(dataset)
    adj_list_train, adj_list_tandv, adj_list_tavat, test_candidate, n_user, n_item = data_partition_amz(dataset)
    statistic_dataset(adj_list_tavat)

    print('n_user', n_user, 'n_item', n_item)
    # import matplotlib.pyplot as plt
    degree_list = np.array([len(adj_list_tavat[u]) for u in adj_list_tavat])
    degree_list = degree_list / degree_list.mean()
    print(degree_list.mean(), degree_list.var())
    # plt.hist(x = degree_list, range=(0, 30), bins=30, color='steelblue', edgecolor='black')
    # # plt.hist(x = degree_list, range=(0, 50), bins=50, color='steelblue', edgecolor='black')
    # # plt.hist(x = degree_list, bins=100, color='steelblue', edgecolor='black')
    # # plt.hist(x = degree_list, color='steelblue', edgecolor='black')
    # # plt.show()
