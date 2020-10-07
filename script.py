import time
import logging
from multiprocessing import cpu_count
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

CODE_VERSION = '1007-1140'

TOPK = 5
EPOCH = 30
LR = 0.002
BATCH_SIZE = 2048
NUM_WORKERS_DL = 4 # dataloader workers, 0 for for single process
NUM_WORKERS_SN = 0 # search_ngh workers, 0 for half cpu core, None for single process
if cpu_count() <= 2:
    # Colab mode
    NUM_WORKERS_DL = 0
    NUM_WORKERS_SN = 2

LAM = 1e-4
EDIM = 64
LAYERS = 2
NUM_NEIGHBORS = 20
POS_ENCODER = 'pos' # time, pos, empty
AGG_METHOD = 'attn' # attn, lstm, mean
ATTN_MODE = 'prod' # prod, map
N_HEAD = 4
DROP_OUT = 0.1

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# register logging logger
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
time_line = time.strftime('%Y%m%d_%H:%M', time.localtime(time.time()))
logfile = time_line + '_tgcn4sr.log'
print('logfile', logfile)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%d%b %H:%M')
console_h = logging.StreamHandler()
console_h.setLevel(logging.INFO)
console_h.setFormatter(formatter)
logger.addHandler(console_h)
if torch.cuda.is_available():
    logfile_h = logging.FileHandler(logfile, mode='w')
    logfile_h.setLevel(logging.INFO)
    logfile_h.setFormatter(formatter)
    logger.addHandler(logfile_h)


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
        # logging.info('train loss ' + str(i) + '/' + str(len(data_loader)) + ': ' + str(loss))
        model.zero_grad()
        # time_start = time.time()
        loss.backward()
        # logging.info('loss.backward time:' + str(time.time() - time_start))
        optimizer.step()
        total_loss += loss.cpu().item()
        # t_e = time.time()
        # logging.info('train one step total time:' + str(t_e - t_s))
        if (i + 1) % log_interval == 0:
            logging.info('Train step: ' + str(i+1) + '/' + str(len(data_loader)) + ' - average loss:' + ' ' + str(total_loss / log_interval))
            total_loss = 0
    # logging.info('train loss:' + ' ' + str(total_loss / len(data_loader)))
    model.del_workers()


def evaluate(model, data_loader):
    with torch.no_grad():
        # logging.info('----- start_evaluate -----')
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
        logging.info('evaluate loss:' + str(avg_loss))
        model.del_workers()


def test(model, data_loader, fast_test=False):
    with torch.no_grad():
        logging.info('----- start_test -----')
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
            # logging.info(candidate_ids.shape) # (2048, 101)
            batch_topk_ids = model.get_top_n(user_id, candidate_ids, time_stamp, num_neighbors=NUM_NEIGHBORS, topk=TOPK).cpu().numpy()
            batch_ndcg = ndcg(batch_topk_ids, target_id)
            ndcg_score.append(batch_ndcg)
            for tgt, topk_ids in zip(target_id, batch_topk_ids):
                total += 1
                if tgt in topk_ids:
                    hit += 1
                    # Test
                    if hit > 100 and hit > total / 2:
                        logging.info(str(tgt) + ' - ' + str(topk_ids))
                        logging.info(str(hit) + '/' + str(total))
        ndcg_score = float(np.mean(ndcg_score))
        logging.info('Test hit rage: ' + str(hit) + '/' + str(total) + ', ndcg: ' + str(ndcg_score))
        model.del_workers()


if __name__ == "__main__":
    print('CODE_VERSION: ' + CODE_VERSION)
    adj_list_train, adj_list_tandv, adj_list_tavat, test_candidate, n_user, n_item = data_partition_amz('newAmazon')

    # train_dataset = TrainDataset(adj_list_train, n_user, n_item)
    tandv_dataset = TrainDataset(adj_list_tandv, n_user, n_item)
    valid_dataset = ValidDataset(adj_list_tavat, n_user, n_item)
    test_dataset = TestDataset(adj_list_tavat, test_candidate, n_user, n_item)

    train_data_loader = DataLoader(tandv_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)
    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)

    train_ngh_finder = NeighborFinder(adj_list_train, n_user, n_item, True) # Initialize training neighbor finder(use train edges)
    test_ngh_finder = NeighborFinder(adj_list_tandv, n_user, n_item, True) # Initialize test neighbor finder(use train and valid edges)

    if POS_ENCODER == 'pos':
        seq_len = 0
        for u in adj_list_tavat:
            if len(adj_list_tavat[u]) > seq_len:
                seq_len = len(adj_list_tavat[u])
    else:
        seq_len = None

    tgcn_model = TGCN(train_ngh_finder, EDIM, n_user+n_item, 2, device, LAYERS, NUM_WORKERS_SN,
                      pos_encoder=POS_ENCODER, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                      n_head=N_HEAD, drop_out=DROP_OUT, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(params=tgcn_model.parameters(), lr=LR, weight_decay=LAM)

    for epoch_i in range(EPOCH):
        logging.info('Train tgcn - epoch ' + str(epoch_i + 1) + '/' + str(EPOCH))
        train(tgcn_model, train_data_loader, optimizer)
        tgcn_model.ngh_finder = test_ngh_finder
        evaluate(tgcn_model, valid_data_loader)
        test(tgcn_model, test_data_loader, fast_test=True)
        tgcn_model.ngh_finder = train_ngh_finder
        logging.info('--------------------------------------------------')
    logging.info('==================================================')
    # test(data_set, model, test_data_loader)
