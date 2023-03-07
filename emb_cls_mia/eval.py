import argparse
import os
import sys
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dsq_attack import system_attack
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from models import GDClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def undefendtest(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    len_t =  int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
    
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return (losses.avg, infer_np)

def main():
    parser = argparse.ArgumentParser(description='undefend eval')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 7, help = 'num class')
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    
    DATASET_PATH = os.path.join('./data')
    checkpoint_path = './'
    # print(checkpoint_path)
    
    train_data_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'))
    train_label_tr_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'))
    train_data_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'))
    train_label_te_attack = np.load(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'))
    train_data = np.load(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'))
    train_label = np.load(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'))
    test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'))
    test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'))
    ref_data = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'))
    ref_label = np.load(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'))
    all_test_data = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'))
    all_test_label = np.load(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'))
    
    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)

    net = GDClassifier()
    resume = checkpoint_path + '/model_best.pth.tar'
    print('==> Resuming from checkpoint:'+resume)
    assert os.path.isfile(resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(resume, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device, torch.float)

    train_data_tr_attack_tensor = torch.from_numpy(train_data_tr_attack).type(torch.FloatTensor)
    train_label_tr_attack_tensor = torch.from_numpy(train_label_tr_attack).type(torch.LongTensor)
    train_data_te_attack_tensor = torch.from_numpy(train_data_te_attack).type(torch.FloatTensor)
    train_label_te_attack_tensor = torch.from_numpy(train_label_te_attack).type(torch.LongTensor)

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)
    all_test_data_tensor = torch.from_numpy(all_test_data).type(torch.FloatTensor)
    all_test_label_tensor = torch.from_numpy(all_test_label).type(torch.LongTensor)
    
    criterion = nn.CrossEntropyLoss().to(device, torch.float)
    # 获取每一部分数据集的置信度向量
    print("training set")
    train_loss, infer_train_conf = undefendtest(train_data_tensor, train_label_tensor, net, criterion, batch_size)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label)

    print("all test set")
    all_test_loss, infer_all_test_conf = undefendtest(all_test_data_tensor, all_test_label_tensor, net, criterion, batch_size)
    all_test_acc, all_test_conf = print_acc_conf(infer_all_test_conf, all_test_label)

    print("training tr set")
    tr_loss, infer_train_conf_tr = undefendtest(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net, criterion, batch_size)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    print("training te set")
    te_loss, infer_train_conf_te = undefendtest(train_data_te_attack_tensor, train_label_te_attack_tensor, net, criterion, batch_size)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)
    print("test set")
    test_loss, infer_test_conf = undefendtest(test_data_tensor, test_label_tensor, net, criterion, batch_size)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    print("reference set")
    ref_loss, infer_ref_conf = undefendtest(ref_data_tensor, ref_label_tensor, net, criterion, batch_size)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)
    
    # MIA
    print("For comparison on undefend model")
    print("avg acc  on train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc ,tr_acc, te_acc, test_acc, ref_acc))
    print("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))
    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack, infer_ref_conf, ref_label, infer_test_conf, test_label, num_class=num_class, attack_epochs=attack_epochs,batch_size=batch_size)
    
    
if __name__ == '__main__':
    main()