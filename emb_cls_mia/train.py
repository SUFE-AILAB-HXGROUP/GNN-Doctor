import argparse
import os
import shutil
import random
import numpy as np
import sys
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import mkdir_p, AverageMeter, accuracy, print_acc_conf
from models import GDClassifier
from dsq_attack import system_attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
        

def train(train_data, train_labels, model, criterion, optimizer, batch_size):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    len_t = int(np.ceil(len(train_data)/batch_size))

    for batch_ind in range(len_t):
        end_idx = min((batch_ind+1)*batch_size, len(train_data))
        inputs = train_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = train_labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg)

def train_eval(test_data, labels, model, criterion, batch_size):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    confs = AverageMeter()

    len_t = int(np.ceil(len(test_data)/batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        # measure data loading time
        end_idx = min(len(test_data), (batch_ind+1)*batch_size)
        inputs = test_data[batch_ind*batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind*batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        infer_np[batch_ind*batch_size: end_idx] = (F.softmax(outputs,dim=1)).detach().cpu().numpy()
        conf = np.mean(np.max(infer_np[batch_ind*batch_size:end_idx], axis = 1))
        confs.update(conf, inputs.size()[0])        

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])

    return (losses.avg, top1.avg, confs.avg)


def main():
    parser = argparse.ArgumentParser(description='undefend training')
    parser.add_argument('--attack_epochs', type = int, default = 150, help = 'attack epochs in NN attack')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--num_class', type = int, default = 7, help = 'num class')
    parser.add_argument('--classifier_epochs', type = int, default = 200, help = 'classifier epochs')

    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    batch_size = args.batch_size
    num_class = args.num_class
    attack_epochs = args.attack_epochs
    classifier_epochs = args.classifier_epochs

    DATASET_PATH = './data'
    checkpoint_path = './'
    #print(checkpoint_path)
    
    # 读取并划分数据
    DATASET_FEATURES = os.path.join(DATASET_PATH,'embeddings.npy')
    DATASET_LABELS = os.path.join(DATASET_PATH,'labels.npy')

    X = np.load('./data/embeddings.npy')
    Y = np.load(DATASET_LABELS)
    
    len_train =len(Y)
    np.random.seed(0)
    r = np.arange(len_train)
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]
    
    # 设置训练数据在总体数据中的比例，及攻击者已知训练数据的比例
    train_classifier_ratio, train_attack_ratio = 0.5, 0.25
    train_data = X[:int(train_classifier_ratio*len_train)]
    ref_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    all_test_data = X[int(train_classifier_ratio*len_train):]

    train_label = Y[:int(train_classifier_ratio*len_train)]
    ref_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    all_test_label = Y[int(train_classifier_ratio*len_train):]

    np.random.seed(1000)
    r = np.arange(len(train_data))
    np.random.shuffle(r)
    train_data_tr_attack = train_data[r[:int(0.5*len(r))]]
    train_label_tr_attack = train_label[r[:int(0.5*len(r))]]

    train_data_te_attack = train_data[r[int(0.5*len(r)):]]
    train_label_te_attack = train_label[r[int(0.5*len(r)):]]

    print(train_data.shape)
    print(test_data.shape)

    print(train_label_tr_attack[:20])
    print(train_label_te_attack[:20])
    print(test_label[:20])
    print(ref_label[:20])
    
    path2 = os.path.join(DATASET_PATH, 'partition')
    if not os.path.isdir(path2):
        mkdir_p(path2)
        
    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_data.npy'), train_data_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'tr_label.npy'), train_label_tr_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_data.npy'), train_data_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'te_label.npy'), train_label_te_attack)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_data.npy'), train_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'train_label.npy'), train_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'ref_data.npy'), ref_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'ref_label.npy'), ref_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_data.npy'), test_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'test_label.npy'), test_label)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_data.npy'), all_test_data)
    np.save(os.path.join(DATASET_PATH, 'partition', 'all_test_label.npy'), all_test_label)
    
    # 训练模型
    model = GDClassifier()
    model = model.to(device,torch.float)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
    train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)

    saved_epoch = 0
    best_acc = 0.0
    
    for epoch in range(1, classifier_epochs+1):
        r= np.arange(len(train_data))
        np.random.shuffle(r)
        train_data = train_data[r]
        train_label = train_label[r]
        train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

        train_loss, trainning_acc = train(train_data_tensor, train_label_tensor, model, criterion, optimizer, batch_size)

        train_loss, train_acc, train_conf = train_eval(train_data_tensor, train_label_tensor, model, criterion, batch_size)
        test_loss, test_acc, test_conf = train_eval(test_data_tensor,test_label_tensor, model, criterion, batch_size)

        # save model
        is_best = test_acc>best_acc
        best_acc = max(test_acc, best_acc)

        if is_best:
            saved_epoch = epoch
            save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'acc': test_acc,
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                    },is_best, checkpoint="./", filename="model_best.pth.tar")

        print('Epoch: [{:d} | {:d}]: acc: training|train|test: {:.4f}|{:.4f}|{:.4f}. conf: train|test: {:.4f}|{:.4f}'.format(epoch, classifier_epochs, trainning_acc, train_acc, test_acc, train_conf, test_conf))
        sys.stdout.flush()

    print("Final saved epoch {:d} acc: {:.4f}.".format(saved_epoch, best_acc))
    
    
if __name__ == '__main__':
    main()