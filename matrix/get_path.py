import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import time
from utils.misc import MetricLogger, batch_device, RAdam
from utils.lr_scheduler import get_linear_schedule_with_warmup
from data import load_data
from model import GraphReasoningModel
from predict import validate
from produce import find_path
from transformers import AdamW
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1)  # avoid using multiple cpus


def get_path_from_ckpt(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ent2id, rel2id, triples, train_loader, test_loader = load_data(args.input_dir, args.bert_name, args.batch_size)
    logging.info("Create model.........")
    model = GraphReasoningModel(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)
    # logging.info(model)

    acc = find_path(args, model, test_loader, device, triples)

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', required=True, help='path to ckpt')
    # training parameters
    parser.add_argument('--bert_lr', default=1e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--opt', default='radam', type=str)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    # model parameters
    parser.add_argument('--bert_name', default='bert-base-uncased',
                        choices=['roberta-base', 'bert-large-uncased', 'bert-base-uncased', 'roberta-large'])
    args = parser.parse_args()
    args.bert_name = args.bert_name
    args.device = os.environ["CUDA_VISIBLE_DEVICES"]
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.file_name = time_ + '_{}_{}_{}_{}'.format(args.opt, args.lr, args.batch_size, args.num_epoch)
    args.model_save_dir = args.save_dir + '/' + args.file_name

    # make logging.info display into both shell and file
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.model_save_dir, args.file_name + ".log"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    get_path_from_ckpt(args)


if __name__ == '__main__':
    main()