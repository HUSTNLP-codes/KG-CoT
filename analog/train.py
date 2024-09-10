import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from utils.misc import MetricLogger, batch_device, RAdam

from data import load_data
from model import GraphReasoningModel
from predict import validate

from transformers import AdamW, get_linear_schedule_with_warmup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1) # avoid using multiple cpus


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ent2id, rel2id, train_loader, test_loader = load_data(args.input_dir, args.bert_name, args.batch_size, args.rev)
    logging.info("Create model.........")
    model = GraphReasoningModel(args, ent2id, rel2id)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    logging.info(model)


    t_total = len(train_loader) * args.num_epoch
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
    print('number of bert param: {}'.format(len(bert_param)))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr': args.bert_lr},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr': args.lr},
        ]

    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    elif args.opt == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        raise NotImplementedError

    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logging.info("Start training........")
    max_acc = 0
    max_epoch = 0


    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            loss = model(*batch_device(batch, device))
            optimizer.zero_grad()
            if isinstance(loss, dict):
                if len(loss) > 1:
                    total_loss = sum(loss.values())
                else:
                    total_loss = loss[list(loss.keys())[0]]
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()

            if iteration % (len(train_loader) // 10) == 0:
            # if True:

                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        test_acc, _ = validate(args, model, test_loader, device)
        logging.info('test acc: {:.4f}'.format(test_acc))
        if (epoch + 1) % 6 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.model_save_dir, 'model-{}-{:.4f}.pt'.format(epoch, test_acc)))
        if max_acc < test_acc:
            if os.path.exists(
                    os.path.join(args.model_save_dir, 'model-{}-{:.4f}_max.pt'.format(max_epoch, max_acc))):
                os.remove(os.path.join(args.model_save_dir, 'model-{}-{:.4f}_max.pt'.format(max_epoch, max_acc)))
            max_acc = test_acc
            max_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(args.model_save_dir, 'model-{}-{:.4f}_max.pt'.format(max_epoch, max_acc)))
    if os.path.exists(os.path.join(args.model_save_dir, 'model-{}-{:.4f}.pt'.format(max_epoch, max_acc))):
        os.remove(os.path.join(args.model_save_dir, 'model-{}-{:.4f}.pt'.format(max_epoch, max_acc)))

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=90, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    # model parameters
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--rev', action='store_true', default='True', help='whether add reversed relations')
    parser.add_argument('--num_ways', default=1, type=int)
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-cased', 'bert-base-uncased'])   # bert-base-cased
    args = parser.parse_args()

    # make logging.info display into both shell and file

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.file_name = time_ + '_{}_{}_{}_{}'.format(args.opt, args.lr, args.batch_size, args.num_epoch)
    args.model_save_dir = args.save_dir + '/' + args.file_name

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    fileHandler = logging.FileHandler(os.path.join(args.model_save_dir, args.file_name + ".log"))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
