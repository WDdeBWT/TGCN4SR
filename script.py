import os
import time
import logging
from multiprocessing import cpu_count
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TGCN
from metrics import ndcg
from graph import NeighborFinder
from data import data_partition_amz, TrainDataset, ValidDataset, TestDataset
from global_flag import flag_true, flag_false

CODE_VERSION = '1207-1031'
LOAD_VERSION = None # '1105-2000' for Amazon
SAVE_CHECKPT = False

DATASET = 'newAmazon' # newAmazon, goodreads_large
TOPK = 5
EPOCH = 20
LR = 0.001
BATCH_SIZE = 4096
NUM_WORKERS_DL = 0 # dataloader workers, 0 for for single process
NUM_WORKERS_SN = 0 # search_ngh workers, 0 for half cpu core, None for single process
USE_MEM = False
if cpu_count() <= 4:
    NUM_WORKERS_SN = cpu_count()
    USE_MEM = True

LAM = 1e-4
FEATURE_DIM = 64
EDGE_DIM = 8
TIME_DIM = 32
LAYERS = 2
NUM_NEIGHBORS = 20
POS_ENCODER = 'pos' # time, pos, empty
AGG_METHOD = 'attn' # attn, lstm, mean, mix
TARGET_MODE = 'prod' # prod, dist
MARGIN = 10
N_HEAD = 4
DROP_OUT = 0.1
USE_TD = True # use time_diff
SA_LAYERS = 0 # self_attn layers
UNIFORM = False
PRUNE = False
if DATASET == 'newAmazon':
    MIN_TRAIN_SEQ = 5
elif DATASET == 'goodreads_large':
    MIN_TRAIN_SEQ = 8
else:
    MIN_TRAIN_SEQ = 8


# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# register logging logger
logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)
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
    logfile_h.setLevel(logging.DEBUG)
    logfile_h.setFormatter(formatter)
    logger.addHandler(logfile_h)


def train(model, data_loader, optimizer, log_interval=100):
    time_start = time.time()
    model.train()
    model.init_workers()
    total_loss = 0
    time_one_interval = time.time()
    # for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(tqdm.tqdm(data_loader)):
    for i, (user_id, pos_id, neg_id, time_stamp) in enumerate(data_loader):
        user_id = user_id.numpy()
        pos_id = pos_id.numpy()
        neg_id = neg_id.numpy()
        time_stamp = time_stamp.numpy()
        loss = model.bpr_loss(user_id, pos_id, neg_id, time_stamp, num_neighbors=NUM_NEIGHBORS)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().item()
        flag_false()
        if (i + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            d_time = time.time() - time_one_interval
            logging.info('Train step: ' + str(i+1) + '/' + str(len(data_loader)) + ' - avg loss: ' + '%.3f' % avg_loss + ' - time: ' + '%.2f' % d_time + 's')
            time_one_interval = time.time()
            total_loss = 0
            flag_true()
    model.del_workers()
    total_time = time.time() - time_start
    logging.info('Train one epoch time: ' + '%.2f' % total_time + 's')


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
        logging.info('evaluate loss: ' + '%.3f' % avg_loss)
        model.del_workers()


def test(model, data_loader, fast_test=1):
    with torch.no_grad():
        logging.info('----- start_test -----')
        model.eval()
        model.init_workers()
        hit = 0
        total = 0
        ndcg_score = []
        for i, (user_id, target_id, candidate_ids, time_stamp) in enumerate(tqdm.tqdm(data_loader)):

            if fast_test != 1:
                cut_len = len(user_id) // fast_test
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
        ndcg_score = float(np.mean(ndcg_score))
        logging.info('Test hit rage: ' + str(hit) + '/' + str(total) + ', ndcg: ' + '%.4f' % ndcg_score)
        model.del_workers()
    return ndcg_score


def load_checkpoint(model, file_path):
    logging.info('Use checkpoint')
    saved_file = torch.load(file_path)
    current_hyper_p = {
        'DATASET': DATASET,
        'LAM': LAM,
        'FEATURE_DIM': FEATURE_DIM,
        'EDGE_DIM': EDGE_DIM,
        'TIME_DIM': TIME_DIM,
        'LAYERS': LAYERS,
        'NUM_NEIGHBORS': NUM_NEIGHBORS,
        'POS_ENCODER': POS_ENCODER,
        'AGG_METHOD': AGG_METHOD,
        'TARGET_MODE': TARGET_MODE,
        'MARGIN': MARGIN,
        'N_HEAD': N_HEAD,
        'DROP_OUT': DROP_OUT,
        'USE_TD': USE_TD,
        'SA_LAYERS': SA_LAYERS,
        'UNIFORM': UNIFORM,
        'MIN_TRAIN_SEQ': MIN_TRAIN_SEQ,
    }
    flag = True
    for key in current_hyper_p:
        if current_hyper_p[key] != saved_file[key]:
            logging.info(key + ' key diff, crt: ' + str(current_hyper_p[key]) + ' - svd: ' + str(saved_file[key]))
            flag = False
    if flag:
        logging.info('All Hyper parameters are same as saved')
    model.load_state_dict(saved_file['state_dict'])


if __name__ == "__main__":
    print('CODE_VERSION: ' + CODE_VERSION)
    adj_list_train, adj_list_tandv, adj_list_tavat, test_candidate, n_user, n_item = data_partition_amz(DATASET)

    # train_dataset = TrainDataset(adj_list_train, n_user, n_item, MIN_TRAIN_SEQ)
    tandv_dataset = TrainDataset(adj_list_tandv, n_user, n_item, MIN_TRAIN_SEQ)
    valid_dataset = ValidDataset(adj_list_tavat, n_user, n_item)
    test_dataset = TestDataset(adj_list_tavat, test_candidate, n_user, n_item)

    train_data_loader = DataLoader(tandv_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)
    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS_DL)

    train_ngh_finder = NeighborFinder(adj_list_train, n_user, n_item, uniform=UNIFORM, use_mem=USE_MEM) # Initialize training neighbor finder(use train edges)
    test_ngh_finder = NeighborFinder(adj_list_tandv, n_user, n_item, uniform=UNIFORM, use_mem=USE_MEM) # Initialize test neighbor finder(use train and valid edges)

    if POS_ENCODER == 'pos':
        seq_len = 0
        for u in adj_list_tavat:
            if len(adj_list_tavat[u]) > seq_len:
                seq_len = len(adj_list_tavat[u])
    else:
        seq_len = None

    tgcn_model = TGCN(train_ngh_finder, FEATURE_DIM, EDGE_DIM, TIME_DIM, n_user+n_item, 2, device,
                    LAYERS, USE_TD, TARGET_MODE, MARGIN, PRUNE, NUM_WORKERS_SN, pos_encoder=POS_ENCODER,
                    agg_method=AGG_METHOD, n_head=N_HEAD, drop_out=DROP_OUT,
                    seq_len=seq_len, sa_layers=SA_LAYERS, data_set=DATASET).to(device)
    optimizer = torch.optim.Adam(params=tgcn_model.parameters(), lr=LR, weight_decay=LAM)

    if LOAD_VERSION is not None:
        load_checkpoint(tgcn_model, LOAD_VERSION + '-' + DATASET + '.pkl')
        tgcn_model.ngh_finder = test_ngh_finder
        test(tgcn_model, test_data_loader, fast_test=10)
        tgcn_model.ngh_finder = train_ngh_finder

    for epoch_i in range(EPOCH):
        logging.info('Train tgcn - epoch ' + str(epoch_i + 1) + '/' + str(EPOCH))
        train(tgcn_model, train_data_loader, optimizer)
        tgcn_model.ngh_finder = test_ngh_finder
        evaluate(tgcn_model, valid_data_loader)
        ndcg_score = test(tgcn_model, test_data_loader, fast_test=3)

        if DATASET == 'newAmazon':
            if ndcg_score > 0.248:
                logging.info('NDCG > 0.248, do 1/3 retest')
                ndcg_score = test(tgcn_model, test_data_loader, fast_test=2)
            if ndcg_score > 0.255:
                logging.info('NDCG > 0.255, do full retest')
                test(tgcn_model, test_data_loader)
        else:
            if ndcg_score > 0.44:
                logging.info('NDCG > 0.44, do 1/3 retest')
                ndcg_score = test(tgcn_model, test_data_loader, fast_test=2)
            if ndcg_score > 0.46:
                logging.info('NDCG > 0.46, do full retest')
                test(tgcn_model, test_data_loader) 

        tgcn_model.ngh_finder = train_ngh_finder
        logging.info('--------------------------------------------------')
    logging.info('==================================================')
    if SAVE_CHECKPT:
        file_to_save = {
            'state_dict': tgcn_model.state_dict(),
            'DATASET': DATASET,
            'LAM': LAM,
            'FEATURE_DIM': FEATURE_DIM,
            'EDGE_DIM': EDGE_DIM,
            'TIME_DIM': TIME_DIM,
            'LAYERS': LAYERS,
            'NUM_NEIGHBORS': NUM_NEIGHBORS,
            'POS_ENCODER': POS_ENCODER,
            'AGG_METHOD': AGG_METHOD,
            'TARGET_MODE': TARGET_MODE,
            'MARGIN': MARGIN,
            'N_HEAD': N_HEAD,
            'DROP_OUT': DROP_OUT,
            'USE_TD': USE_TD,
            'SA_LAYERS': SA_LAYERS,
            'UNIFORM': UNIFORM,
            'MIN_TRAIN_SEQ': MIN_TRAIN_SEQ,
        }
        save_path = CODE_VERSION + '-' + DATASET + '.pkl'
        torch.save(file_to_save, save_path)
    tgcn_model.ngh_finder = test_ngh_finder
    test(tgcn_model, test_data_loader, fast_test=5)
