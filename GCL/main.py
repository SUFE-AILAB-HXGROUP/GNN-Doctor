from torch_geometric import seed_everything
import warnings
import argparse
import torch
import os
from GCL_node import *
import logging
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument("--method", type=str, default='GCA', help="{DGI_inductive, DGI_transductive, MVGRL, GRACE, GCA}")
    parser.add_argument("--drop_scheme", type=str, default='degree', help="{degree, evc, pr, uniform}, used for GCA")
    parser.add_argument("--dataset", type=str, default='Cora', help="dataset name.")
    parser.add_argument("--path", type=str, default='data/', help="folder of dataset file.")
    parser.add_argument("--seed", type=int, default=42, help='random seed.')
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    args = parser.parse_args()

    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    if args.method == 'GRACE':
        args.drop_scheme = 'uniform'

    return args


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)
    return logger


if __name__ == '__main__':
    args = get_parser()
    seed_everything(args.seed)

    if not os.path.exists('log/'):
        os.makedirs('log/')

    logger = config_logger('log/' + args.method + '_' + args.dataset + '.log')
    logger.info(args)

    if args.method in ['GCA', 'GRACE']:
        GCA(args, logger)
    elif args.method == 'MVGRL':
        MVGRL(args, logger)
    elif args.method == 'DGI_inductive':
        DGI_inductive(args, logger)
    elif args.method == 'DGI_transductive':
        DGI_transductive(args, logger)
    else:
        raise NotImplementedError


