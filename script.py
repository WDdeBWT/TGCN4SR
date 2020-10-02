import time
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TGCN
from metrics import ndcg
from graph import NeighborFinder
from data import data_partition_amz, TrainDataset, ValidDataset, TestDataset

EPOCH = 20
LR = 0.01
EDIM = 64
LAYERS = 2
LAM = 1e-4
NUM_NEIGHBORS = 20
TOPK = 5

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data_loader, optimizer, log_interval=50):
    model.train()
    model.init_workers()
    total_loss = 0
    for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(tqdm.tqdm(data_loader)):
    # for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(data_loader):
        # t_s = time.time()
        user_id = user_id.numpy()
        pos_id = pos_id.numpy()
        neg_id = neg_id.numpy()
        time_stamp = time_stamp.numpy()
        loss = model.bpr_loss(user_id, pos_id, neg_id, time_stamp, num_neighbors=NUM_NEIGHBORS)
        # print('train loss ' + str(i) + '/' + str(len(data_loader)) + ': ' + str(loss))
        model.zero_grad()
        # time_start = time.time()
        loss.backward()
        # print('loss.backward time:' + str(time.time() - time_start))
        optimizer.step()
        total_loss += loss.cpu().item()
        # t_e = time.time()
        # print('train one step total time:', t_e - t_s)
        if (i + 1) % log_interval == 0:
            print('    - Average loss:', total_loss / log_interval)
            total_loss = 0
    # print('train loss:', total_loss / len(data_loader))
    model.del_workers()


def evaluate(model, data_loader):
    with torch.no_grad():
        # print('----- start_evaluate -----')
        model.eval()
        model.init_workers()
        total_loss = 0
        # for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(tqdm.tqdm(data_loader)):
        for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(data_loader):
            user_id = user_id.numpy()
            pos_id = pos_id.numpy()
            neg_id = neg_id.numpy()
            time_stamp = time_stamp.numpy()
            loss = model.bpr_loss(user_id, pos_id, neg_id, time_stamp, num_neighbors=NUM_NEIGHBORS)
            total_loss += loss.cpu().item()
        avg_loss = total_loss / len(data_loader)
        print('evaluate loss:' + str(avg_loss))
        model.del_workers()

def test(model, data_loader, fast_test=False):
    with torch.no_grad():
        # print('----- start_test -----')
        model.eval()
        model.init_workers()
        hit = 0
        total = 0
        ndcg_score = []
        for i, (user_id, target_id, candidate_ids, time_stamp) in enumerate(tqdm.tqdm(data_loader)):

            if fast_test:
                cut_len = len(user_id) // 10
                user_id = user_id[:cut_len]
                target_id = target_id[:cut_len]
                candidate_ids = candidate_ids[:cut_len]
                time_stamp = time_stamp[:cut_len]

            user_id = user_id.numpy()
            target_id = target_id.numpy()
            candidate_ids = candidate_ids.numpy()
            time_stamp = time_stamp.numpy()
            # print(candidate_ids.shape) # (2048, 101)
            batch_topk_ids = model.get_top_n(user_id, candidate_ids, time_stamp, num_neighbors=NUM_NEIGHBORS, topk=TOPK).cpu().numpy()
            batch_ndcg = ndcg(batch_topk_ids, target_id)
            ndcg_score.append(batch_ndcg)
            for tgt, topk_ids in zip(target_id, batch_topk_ids):
                total += 1
                if tgt in topk_ids:
                    hit += 1
                    # Test
                    # if hit > 100 and hit > total / 2:
                    #     print(tgt, topk_ids)
                    #     print(hit, total)
        ndcg_score = float(np.mean(ndcg_score))
        print('Test hit rage: ' + str(hit) + '/' + str(total) + ', ndcg: ' + str(ndcg_score))
        model.del_workers()


if __name__ == "__main__":
    adj_list_train, adj_list_tandv, adj_list_tavat, test_candidate, n_user, n_item = data_partition_amz('newAmazon')

    # train_dataset = TrainDataset(adj_list_train, n_user, n_item)
    tandv_dataset = TrainDataset(adj_list_tandv, n_user, n_item)
    valid_dataset = ValidDataset(adj_list_tavat, n_user, n_item)
    test_dataset = TestDataset(adj_list_tavat, test_candidate, n_user)

    train_data_loader = DataLoader(tandv_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_data_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True, num_workers=4)

    train_ngh_finder = NeighborFinder(adj_list_train, n_user, n_item, True) # Initialize training neighbor finder(use train edges)
    test_ngh_finder = NeighborFinder(adj_list_tandv, n_user, n_item, True) # Initialize test neighbor finder(use train and valid edges)

    tgcn_model = TGCN(train_ngh_finder, EDIM, n_user+n_item, 2, device, LAYERS).to(device)
    optimizer = torch.optim.Adam(params=tgcn_model.parameters(), lr=LR, weight_decay=LAM)

    for epoch_i in range(EPOCH):
        print('Train tgcn - epoch ' + str(epoch_i + 1) + '/' + str(EPOCH))
        train(tgcn_model, train_data_loader, optimizer)
        # tgcn_model.ngh_finder = test_ngh_finder
        evaluate(tgcn_model, valid_data_loader, test_ngh_finder)
        test(tgcn_model, test_data_loader, test_ngh_finder, fast_test=True)
        # tgcn_model.ngh_finder = train_ngh_finder
        # if (epoch_i + 1) % 10 == 0:
        #     test(data_set, model, test_data_loader)
        print('--------------------------------------------------')
    print('==================================================')
    # test(data_set, model, test_data_loader)
