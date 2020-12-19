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

    def count_item(_path_to_data, _item_count):
        f = open(_path_to_data, 'r')
        for line in f:
            try:
                u, i, t = line.rstrip().split('\t')
                try:
                    u = int(u)
                    i = int(i)
                    t = int(t)
                except:
                    u = int(u)
                    i = int(i)
                    t = int(float(t))
            except:
                u, i, r, t = line.rstrip().split('\t')
                try:
                    u = int(u)
                    i = int(i)
                    t = int(t)
                except:
                    u = int(u)
                    i = int(i)
                    t = int(float(t))
            _item_count[i] += 1
        f.close()

    def count_user(_path_to_data, _item_count, _user_count):
        # should after count item
        f = open(_path_to_data, 'r')
        for line in f:
            try:
                u, i, t = line.rstrip().split('\t')
                try:
                    u = int(u)
                    i = int(i)
                    t = int(t)
                except:
                    u = int(u)
                    i = int(i)
                    t = int(float(t))
            except:
                u, i, r, t = line.rstrip().split('\t')
                try:
                    u = int(u)
                    i = int(i)
                    t = int(t)
                except:
                    u = int(u)
                    i = int(i)
                    t = int(float(t))
            if _item_count[i] < 5:
                continue
            _user_count[u] += 1
        f.close()

    data_path = datastore_path + '/' + dataset_name + '/'

    item_count = defaultdict(lambda: 0)
    user_count = defaultdict(lambda: 0)
    count_item(data_path + dataset_name + '.txt', item_count)
    count_user(data_path + dataset_name + '.txt', item_count, user_count)

    f = open(data_path + dataset_name + '.txt', 'r')
    user_dict = defaultdict(list)
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    for line in f:
        try:
            u, i, t = line.rstrip().split('\t')
            try:
                u = int(u)
                i = int(i)
                t = int(t)
            except:
                u = int(u)
                i = int(i)
                t = int(float(t))
        except:
            u, i, r, t = line.rstrip().split('\t')
            try:
                u = int(u)
                i = int(i)
                t = int(t)
            except:
                u = int(u)
                i = int(i)
                t = int(float(t))
        if item_count[i] < 5 or user_count[u] < 5:
            continue

        if u in usermap:
            userid = usermap[u]
        else:
            usernum += 1
            userid = usernum
            usermap[u] = userid
        if i in itemmap:
            itemid = itemmap[i]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[i] = itemid
        user_dict[userid].append([itemid, t])
    f.close()
    print(len(user_dict))

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
