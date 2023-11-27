import numpy as np
from tqdm import tqdm
from amazon import AmazonDataset
from DMRL_base import DMRL_Base
import argparse
from time import time
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import create_logger
import shutil
from evaluator20_accelerate import RecallEvaluator

def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch

def train(model, optimizer, train_loader, train_dataset, test_dataset, train_num, logger, log_path):
    EVALUATION_EVERY_N_BATCHES = train_num // args.batch_size + 1
    cur_best_pre_0 = 0.
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    stopping_step = 0
    for epoch_num in range(args.epochs):
        t1 = time()
        # TODO: early stopping based on validation recall
        # train model
        losses = 0
        model.train()
        # run n mini-batches
        for _ in tqdm(range(10), desc="Model train"):
            for b, batch in enumerate(train_loader):
                batch = to_cuda(batch)
                optimizer.zero_grad()
                loss = model(*batch)
                loss.backward()
                optimizer.step()
                losses += loss
        t2 = time()
        model.eval()
        testresult = RecallEvaluator(model, train_dataset, test_dataset)
        recalls, precisions, hit_ratios, ndcgs = testresult.eval(model)
        rec_loger.append(recalls)
        pre_loger.append(precisions)
        ndcg_loger.append(ndcgs)
        hit_loger.append(hit_ratios)
        t3 = time()
        print("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (
        epoch_num, t2 - t1, t3 - t2, losses / (10 * EVALUATION_EVERY_N_BATCHES), recalls, precisions, hit_ratios, ndcgs))
        logger.info("epochs%d  [%.1fs + %.1fs]: train loss=%.5f, result=recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (
        epoch_num, t2 - t1, t3 - t2, losses / (10 * EVALUATION_EVERY_N_BATCHES), recalls, precisions, hit_ratios, ndcgs))

        cur_best_pre_0, stopping_step, should_stop = early_stopping(recalls, cur_best_pre_0, stopping_step, model, optimizer, log_path,
                                                                    expected_order='acc', flag_step=5)    # TODO: 受epoch影响
        if should_stop == True:
            break
        if epoch_num == args.epochs - 1:
            break
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs)
    idx = list(recs).index(best_rec_0)
    final_perf = "Best Iter = recall:%.5f, pres:%.5f, hr:%.5f, ndcg:%.5f" % (recs[idx], pres[idx], hit[idx], ndcgs[idx])
    print(final_perf)
    logger.info(final_perf)


def early_stopping(log_value, best_value, stopping_step, model, optimizer, log_path, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = model.state_dict()
        checkpoint_dict['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint_dict, os.path.join(log_path, 'best_param.model'))
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop

def parse_args():
    parser = argparse.ArgumentParser(description='Run DMRL.')
    parser.add_argument('--dataset', nargs='?',default='Office', help='Choose a dataset.')
    parser.add_argument('--epochs', type=int,default=1000, help = 'total_epochs')
    parser.add_argument('--gpu', nargs='?',default='0', help = 'gpu_id')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate.')
    parser.add_argument('--decay_r', type=float, default=1e-0, help='decay_r.')
    parser.add_argument('--decay_c', type=float, default=1e-3, help='decay_c.')
    parser.add_argument('--decay_p', type=float, default=0, help='decay_p.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--n_factors', type=int, default=4,help='Number of factors.')
    parser.add_argument('--num_neg', type=int,default=4, help = 'negative items')
    parser.add_argument('--hidden_layer_dim_a', type=int, default=256, help='Hidden layer dim a.')
    parser.add_argument('--hidden_layer_dim_b', type=int, default=128, help='Hidden layer dim b.')
    parser.add_argument('--dropout_a', type=float, default=0.2, help='dropout_a.')
    parser.add_argument('--dropout_b', type=float, default=0.2, help='dropout_b.')
    parser.add_argument('--emb_dim', type=int, default=128, help='emb_dim.')
    parser.add_argument('--layer', type=int,default=3, help="the layer num of GCN")
    parser.add_argument('--is_dropout', action='store_true', default=False, help='whether need to dropout the adj matrix')
    parser.add_argument('--keepprob', type=float, default=0.6, help='the batch size for bpr loss training procedure')
    parser.add_argument('--n_folds', type=int,default=0, help="the fold num used to split large adj matrix, like Clothing")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay.')
    args = parser.parse_args()

    return args

def mainer(args, dataset, num_neg):
    args.dataset = dataset
    args.num_neg = num_neg
    print(args)

    Filename = dataset
    Filepath = 'AmazonData/' + Filename
    log_path = f'Result/{dataset}/0226/loss_xavier_sum_wo-scheduler_model加l2norm_reshape改变_unfix_neg{str(args.num_neg)}_embdim{str(args.emb_dim)}_hiddena{str(args.hidden_layer_dim_a)}_hiddenb{str(args.hidden_layer_dim_b)}_lr{str(int(args.learning_rate*10000))}_factor{str(args.n_factors)}_decayr{str(int(args.decay_r*100000))}_decayc{str(int(args.decay_c*100000))}_weightdecay{str(int(args.weight_decay*100000))}'
    logger = create_logger(log_path)
    logger.info('\nLog path is: ' + log_path)
    logger.info('training args:{}'.format(args))
    shutil.copy('main.py', log_path)
    shutil.copy('DMRL_base.py', log_path)
    shutil.copy('amazon.py', log_path)

    train_dataset = AmazonDataset(Filepath, 'train', args.dataset, n_negative=args.num_neg, n_folds = args.n_folds)
    test_dataset = AmazonDataset(Filepath, 'test', args.dataset, n_negative=args.num_neg, n_folds = 0)
    n_users, n_items = train_dataset.n_users, train_dataset.m_items
    print(n_users, n_items)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=6, drop_last=True)
    # test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=6)

    model = DMRL_Base(n_users,
                 n_items,
                 train_dataset,
                 num_neg=args.num_neg,
                 n_factors=args.n_factors,
                 embed_dim=args.emb_dim,
                 decay_r=args.decay_r,
                 decay_c=args.decay_c,
                 hidden_layer_dim_a=args.hidden_layer_dim_a,
                 hidden_layer_dim_b=args.hidden_layer_dim_b,
                 dropout_rate_a=args.dropout_a,
                 dropout_rate_b=args.dropout_b,
                 dataset_name=args.dataset,
                 layer=args.layer,
                 keep_prob = args.keepprob,
                 is_dropout = args.is_dropout,
                 A_split = args.A_split
                 )
    num_gpus = len(args.gpu.split(','))
    model = torch.nn.DataParallel(model).cuda() if num_gpus > 1 else model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(m.weight)

    train(model, optimizer, train_loader, train_dataset, test_dataset, train_dataset.data_num, logger, log_path)

if __name__ == '__main__':
    seed = 12345
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # get user-item matrix
    # make feature as dense matrix
    args = parse_args()
    dataset = args.dataset
    args.A_split = False
    if args.n_folds > 0:
        args.A_split = True
    # dataset = 'Baby'
    # emb_dim = 128
    num_neg = args.num_neg
    # decay_c = 1e-3
    mainer(args, dataset, num_neg)
