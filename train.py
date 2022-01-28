import argparse
from tqdm import tqdm
from loguru import logger

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import TrainDataset, TestDataset
from model import SimcseModel, simcse_unsup_loss, simcse_sup_loss
from transformers import BertModel, BertConfig, BertTokenizer
import os
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
import pandas as pd
import time
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances


def seed_everything(seed=42):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train(model, train_loader, dev_loader, dev_df, optimizer, args):
    logger.info("start training")
    model.train()
    device = args.device
    best = 0
    for epoch in range(args.epochs):
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            # [batch, n, seq_len] -> [batch * n, sql_len]
            sql_len = data['input_ids'].shape[-1]
            input_ids = data['input_ids'].view(-1, sql_len).to(device)
            attention_mask = data['attention_mask'].view(-1, sql_len).to(device)
            token_type_ids = data['token_type_ids'].view(-1, sql_len).to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            if args.train_mode == 'unsupervise':
                loss = simcse_unsup_loss(out, device)
            else:
                loss = simcse_unsup_loss(out, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.eval_step == 0:
                precision, recall, f1, predictions = evaluate(model, dev_loader, dev_df, device,
                                                              threshold=args.threshold)
                logger.info('loss:{}, f1:{}, precision: {}, recall:{} in step {} epoch {}'.format(
                    loss, f1, precision, recall, step, epoch)
                )
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('f1', f1, step)
                writer.add_scalar('precision', precision, step)
                writer.add_scalar('recall', recall, step)

                model.train()
                if best < f1:
                    best = f1
                    torch.save(model.state_dict(), join(args.output_path, 'simcse.pt'))
                    logger.info('higher f1: {} in step {} epoch {}, save model'.format(best, step, epoch))


def evaluate(model, dataloader, df, device, threshold=0.5):
    model.eval()

    embeddings = torch.tensor([], device=device)
    with torch.no_grad():
        for source in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            sql_len = source['input_ids'].shape[-1]
            source_input_ids = source.get('input_ids').view(-1, sql_len).to(device)
            source_attention_mask = source.get('attention_mask').view(-1, sql_len).to(device)
            source_token_type_ids = source.get('token_type_ids').view(-1, sql_len).to(device)
            # pdb.set_trace()
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)

            # embeddings = np.append(embeddings, source_pred.numpy())
            embeddings = torch.cat((embeddings, source_pred), dim=0)

    distances = cosine_distances(embeddings.cpu(), embeddings.cpu())
    distances = torch.from_numpy(distances).to(device)
    distances, indices = torch.sort(distances, dim=1)
    distances = distances.cpu().numpy()
    indices = indices.cpu().numpy()

    # get predictions
    predictions = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    for k in range(embeddings.shape[0]):
        # pdb.set_trace()
        # 第k个数据的label
        label = df['label_group'].iloc[k]
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        # 模型认为相似的title的label
        predict_lst = df['label_group'].iloc[ids].values
        # 预测正确的数量
        num_right = len([x for x in predict_lst if x == label])
        # 该label实际上存在的case的数量
        num_label = len(df.loc[df['label_group'] == label])

        precision = num_right / len(predict_lst)
        recall = num_right / num_label
        f1 = 2 * (precision * recall) / (precision + recall)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)

        posting_ids = np.unique(df['posting_id'].iloc[ids].values)
        predictions.append(posting_ids)

    precision = sum(precision_lst) / len(precision_lst)
    recall = sum(recall_lst) / len(recall_lst)
    f1 = sum(f1_lst) / len(f1_lst)
    return precision, recall, f1, predictions


def load_train_data_unsupervised(tokenizer, args):
    """
    获取无监督训练语料，对于每个title，复制一份作为正样本
    """
    logger.info('loading unsupervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-unsupervise.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    df = pd.read_csv(args.train_file, sep=',')
    rows = df.to_dict('records')
    for row in tqdm(rows):
        title = row['title']
        title_ids = tokenizer([title, title], max_length=args.max_len, truncation=True, padding='max_length',
                              return_tensors='pt')
        feature_list += title_ids

    logger.info("len of train data:{}".format(len(feature_list)))
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_train_data_supervised(tokenizer, args):
    """
    获取有监督训练数据，同一个类别下，两两title组成一条训练数据
    """
    # 加载缓存数据
    logger.info('loading supervised train data')
    output_path = os.path.dirname(args.output_path)
    train_file_cache = join(output_path, 'train-supervised.pkl')
    if os.path.exists(train_file_cache) and not args.overwrite_cache:
        with open(train_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of train data:{}".format(len(feature_list)))
            return feature_list
    feature_list = []
    df = pd.read_csv(args.train_file, sep=',')
    logger.info("len of train data:{}".format(len(df)))
    label2titles = defaultdict(list)
    rows = df.to_dict('records')
    # rows = rows[:10000]

    # 收集每个label_group下的title集合
    for row in tqdm(rows):
        title = row['title']
        label = row['label_group']
        label2titles[label].append(title)

    # todo
    for label, titles in tqdm(label2titles.items()):
        # 同一类别下，两两组成一条训练数据
        titles_tokens = tokenizer(titles, max_length=args.max_len, truncation=True, padding='max_length',
                                  return_tensors='pt')
        for i in range(len(titles)):
            for j in range(len(titles)):
                if i >= j:
                    continue
                input_ids = torch.cat(
                    [titles_tokens['input_ids'][i].unsqueeze(0), titles_tokens['input_ids'][j].unsqueeze(0)], dim=0)
                token_type_ids = torch.cat(
                    [titles_tokens['token_type_ids'][i].unsqueeze(0), titles_tokens['token_type_ids'][j].unsqueeze(0)],
                    dim=0)
                attention_mask = torch.cat(
                    [titles_tokens['attention_mask'][i].unsqueeze(0), titles_tokens['attention_mask'][j].unsqueeze(0)],
                    dim=0)
                feature_list.append(
                    {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask})

    logger.info("len of train data:{}".format(len(feature_list)))
    with open(train_file_cache, 'wb') as f:
        pickle.dump(feature_list, f)
    return feature_list


def load_eval_data(tokenizer, args, mode):
    """
    加载验证集或者测试集
    """
    assert mode in ['dev', 'test'], 'mode should in ["dev", "test"]'
    logger.info('loading {} data'.format(mode))
    output_path = os.path.dirname(args.output_path)
    eval_file_cache = join(output_path, '{}.pkl'.format(mode))
    if os.path.exists(eval_file_cache) and not args.overwrite_cache:
        with open(eval_file_cache, 'rb') as f:
            feature_list = pickle.load(f)
            logger.info("len of {} data:{}".format(mode, len(feature_list)))
            return feature_list

    if mode == 'dev':
        eval_file = args.dev_file
    else:
        eval_file = args.test_file

    df = pd.read_csv(eval_file, sep=',')
    logger.info("len of {} data:{}".format(mode, len(df)))
    feature_list = []
    rows = df.to_dict('records')
    for index, row in enumerate(tqdm(rows)):
        title = row['title']
        feature = tokenizer(title, max_length=args.max_len, truncation=True, padding='max_length',
                            return_tensors='pt')
        feature_list.append(feature)

    res = {'df': df, 'feature_list': feature_list}
    with open(eval_file_cache, 'wb') as f:
        pickle.dump(res, f)
    return res


def main(args):
    # 加载模型
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    assert args.pooler in ['cls', "pooler", "last-avg", "first-last-avg"], \
        'pooler should in ["cls", "pooler", "last-avg", "first-last-avg"]'
    model = SimcseModel(pretrained_model=args.pretrain_model_path, pooling=args.pooler, dropout=args.dropout).to(
        args.device)
    # pdb.set_trace()
    if args.do_train:
        # 加载数据集
        assert args.train_mode in ['supervise', 'unsupervise'], \
            "train_mode should in ['supervise', 'unsupervise']"
        if args.train_mode == 'supervise':
            train_data = load_train_data_supervised(tokenizer, args)
        elif args.train_mode == 'unsupervise':
            train_data = load_train_data_unsupervised(tokenizer, args)
        train_dataset = TrainDataset(train_data, tokenizer, max_len=args.max_len)
        # train_dataset = train_dataset[:128]
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers)
        dev_data = load_eval_data(tokenizer, args, 'dev')
        dev_df = dev_data['df']
        dev_dataset = TestDataset(dev_data['feature_list'], tokenizer, max_len=args.max_len)
        # dev_dataset = dev_dataset[:8]
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train(model, train_dataloader, dev_dataloader, dev_df, optimizer, args)
    if args.do_predict:
        test_data = load_eval_data(tokenizer, args, 'test')
        test_df = test_data['df']
        test_dataset = TestDataset(test_data['feature_list'], tokenizer, max_len=args.max_len)
        # test_dataset = test_dataset[:8]
        # test_df = test_df.iloc()
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers)
        path = join(args.output_path, 'simcse.pt')
        logger.info(path)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(int(model.bert.embeddings.position_ids[0, -1]))
        precision, recall, f1, predictions = evaluate(model, test_dataloader, test_df, args.device,
                                                      threshold=args.threshold)
        logger.info('testset precision:{}, recall:{}, f1:{}'.format(precision, recall, f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
    parser.add_argument("--output_path", type=str, default='output/shopee')
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_eval", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_step", type=int, default=50, help="every eval_step to evaluate model")
    parser.add_argument("--max_len", type=int, default=150, help="max length of input")
    parser.add_argument("--threshold", type=float, default=0.3, help="threshold")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--train_file", type=str, default="data/shopee/train.csv")
    parser.add_argument("--dev_file", type=str, default="data/shopee/dev.csv")
    parser.add_argument("--test_file", type=str, default="data/shopee/test.csv")
    parser.add_argument("--pretrain_model_path", type=str,
                        default="pretrain_model/bert-base-uncased")
    parser.add_argument("--pooler", type=str, choices=['cls', "pooler", "last-avg", "first-last-avg"],
                        default='cls', help='pooler to use')
    parser.add_argument("--train_mode", type=str, default='supervise', choices=['unsupervise', 'supervise'],
                        help="unsupervise or supervise")
    parser.add_argument("--overwrite_cache", action='store_true', default=True, help="overwrite cache")
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--do_predict", action='store_true', default=True)

    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
    args.output_path = join(args.output_path, args.train_mode,
                            'bsz-{}-lr-{}-dropout-{}-threshold-{}'.format(args.batch_size_train, args.lr, args.dropout,
                                                                          args.threshold))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args)


