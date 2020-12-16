import os
import time
from collections import defaultdict

import numpy as np


def get_date(ts):
    timeArray = time.localtime(float(ts))
    ntCtime_str = time.strftime("%m %d, %Y", timeArray)
    # print(ntCtime_str)
    return ntCtime_str


def write_pos_file(datastore_path, dataset_name):
    data_path = datastore_path + '/' + dataset_name + '/'
    f = open(data_path + dataset_name + '_ori.txt', 'r')
    u_set = set()
    i_set = set()
    user_dict = defaultdict(list)
    for line in f:
        u, i, t = line.rstrip().split('\t')
        try:
            u = int(u)
            i = int(i)
            t = int(t)
        except:
            u = int(u)
            i = int(i)
            t = int(float(t))
        u_set.add(u)
        i_set.add(i)
        user_dict[u].append([i, t])
    f.close()

    max_u = max(u_set)
    for u in range(max_u):
        assert u+1 in u_set, str(u+1)
    max_i = max(i_set)
    for i in range(max_i):
        assert i+1 in i_set
    print(max_u, max_i)

    for userid in user_dict.keys():
        user_dict[userid].sort(key=lambda x: x[1])

    f = open(data_path + dataset_name + '_all.txt', 'w')
    for user_id in user_dict.keys():
        for i in user_dict[user_id]:
            f.write(str(user_id) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + get_date(i[1]) + '\n')
    f.close()

    f = open(data_path + dataset_name + '_lite.txt', 'w')
    for user_id in user_dict.keys():
        for i in user_dict[user_id]:
            f.write('%d %d\n' % (user_id, i[0]))
    f.close()


def write_neg_file(datastore_path, dataset_name):
    data_path = datastore_path + '/' + dataset_name + '/'

    adj_list_original = defaultdict(list)
    test_candidate = {}

    # assume user/item index starting from 1
    path_to_data = data_path + dataset_name + '_all.txt'

    max_item = 0
    f = open(path_to_data, 'r')
    for line in f:
        u, i, t, d = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        if i > max_item:
            max_item = i
        adj_list_original[u].append(i)
    user_list = list(adj_list_original.keys())
    user_list.sort()
    f.close()

    write_list = []
    for user_id in user_list:
        temp_neg_list = []
        while len(temp_neg_list) < 100:
            neg_id = np.random.randint(1, max_item)
            if neg_id not in adj_list_original[user_id]:
                temp_neg_list.append(neg_id)
        for neg_id in temp_neg_list:
            write_list.append(str(user_id) + '\t' + str(neg_id) + '\n')
    f = open(data_path + dataset_name + '_test_neg.txt', 'w')
    f.writelines(write_list)
    f.close()


if __name__ == "__main__":
    # Data prepare for TGCN4SR & HyperRec, original data is from TiSASRec
    datastore_path = 'tisasrec_data'
    dataset_name = 'steam'
    write_pos_file(datastore_path, dataset_name)
    write_neg_file(datastore_path, dataset_name)
