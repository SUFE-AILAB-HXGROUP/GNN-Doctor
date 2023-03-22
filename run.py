import argparse
import logging

import torch

from data_utils.data_load import load_dataset
from target_model_utils import target_model_train
from attack_utils import attack_func


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--setting', type=str, default='black_box',
                        help="black-box GNNs or node embedding publishing")
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--attack', type=str, default='link_steal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-model', type=str, default='gcn')
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--attack-epoch', type=int, default=3000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Working device: cuda or cpu
    device = torch.device(args.device)

    # Loading dataset
    dataset = load_dataset(args.dataset, args.setting, args.attack)

    # Training target model
    model = target_model_train(dataset, args.setting, args.target_model, args.epoch, device)

    # Conducting attack
    attack_func(dataset, model, args.setting, args.attack, args.attack_epoch, device)
