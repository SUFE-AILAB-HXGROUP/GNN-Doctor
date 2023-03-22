import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


small_delta = 1e-30


class GDClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(GDClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        out = self.classifier(hidden_out)
        return out


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class InferenceAttack_HZ(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(InferenceAttack_HZ, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, x1, l):
        out_x1 = self.features(x1)
        out_l = self.labels(l)
        is_member = self.combine(torch.cat((out_x1, out_l), 1))
        return self.output(is_member)


class InferenceAttack_HZWL(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(InferenceAttack_HZWL, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()

    def forward(self, x1):
        out_x1 = self.features(x1)
        is_member = self.combine(out_x1)
        return self.output(is_member)


def mia_emb_func(X, y, device):
    # params
    batch_size = 128
    num_class = len(set(y))
    attack_epochs = 150
    classifier_epochs = 200

    len_train = len(y)
    np.random.seed(0)
    r = np.arange(len_train)
    np.random.shuffle(r)
    X = X[r]
    y = y[r]

    # split data
    train_classifier_ratio, train_attack_ratio = 0.5, 0.25
    train_data = X[:int(train_classifier_ratio * len_train)]
    ref_data = X[int(train_classifier_ratio * len_train):int((train_classifier_ratio + train_attack_ratio) * len_train)]
    test_data = X[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    all_test_data = X[int(train_classifier_ratio * len_train):]

    train_label = y[:int(train_classifier_ratio * len_train)]
    ref_label = y[
                int(train_classifier_ratio * len_train):int((train_classifier_ratio + train_attack_ratio) * len_train)]
    test_label = y[int((train_classifier_ratio + train_attack_ratio) * len_train):]
    all_test_label = y[int(train_classifier_ratio * len_train):]

    np.random.seed(1000)
    r = np.arange(len(train_data))
    np.random.shuffle(r)
    train_data_tr_attack = train_data[r[:int(0.5 * len(r))]]
    train_label_tr_attack = train_label[r[:int(0.5 * len(r))]]

    train_data_te_attack = train_data[r[int(0.5 * len(r)):]]
    train_label_te_attack = train_label[r[int(0.5 * len(r)):]]

    # train and save target classifier model
    train_cls_model(device, train_data, train_label, test_data, test_label, classifier_epochs, batch_size)

    # evaluate the mia risk
    eval_mia_risk(train_data, train_label, test_data, test_label, ref_data, ref_label, all_test_data, all_test_label,
                  train_data_tr_attack, train_label_tr_attack, train_data_te_attack, train_label_te_attack,
                  device, batch_size, attack_epochs, num_class)


def train_cls_model(device, train_data, train_label, test_data, test_label, epochs, batch_size):
    model = GDClassifier()
    model = model.to(device, torch.float)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device, torch.float)

    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    saved_epoch = 0
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        r = np.arange(len(train_data))
        np.random.shuffle(r)
        train_data = train_data[r]
        train_label = train_label[r]
        train_data_tensor = torch.from_numpy(train_data).type(torch.FloatTensor)
        train_label_tensor = torch.from_numpy(train_label).type(torch.LongTensor)

        train_loss, trainning_acc = train(train_data_tensor, train_label_tensor, model, criterion, optimizer,
                                          batch_size, device)

        train_loss, train_acc, train_conf = train_eval(train_data_tensor, train_label_tensor, model, criterion,
                                                       batch_size, device)
        test_loss, test_acc, test_conf = train_eval(test_data_tensor, test_label_tensor, model, criterion,
                                                    batch_size, device)

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if is_best:
            saved_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, checkpoint="./", filename="model_best.pth.tar")

        logging.info(
            'Epoch: [{:d} | {:d}]: acc: training|train|test: {:.4f}|{:.4f}|{:.4f}. conf: train|test: '
            '{:.4f}|{:.4f}'.format(epoch, epochs, trainning_acc, train_acc, test_acc, train_conf, test_conf))

    logging.info("Final saved epoch {:d} acc: {:.4f}.".format(saved_epoch, best_acc))


def eval_mia_risk(train_data, train_label, test_data, test_label, ref_data, ref_label, all_test_data, all_test_label,
                  train_data_tr_attack, train_label_tr_attack, train_data_te_attack, train_label_te_attack,
                  device, batch_size, epochs, num_class):
    test_data_tensor = torch.from_numpy(test_data).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_label).type(torch.LongTensor)

    ref_data_tensor = torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor = torch.from_numpy(ref_label).type(torch.LongTensor)

    net = GDClassifier()
    resume = './/model_best.pth.tar'
    logging.info('==> Resuming from checkpoint:' + resume)
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
    logging.info("training set")
    train_loss, infer_train_conf = undefendtest(train_data_tensor, train_label_tensor, net, criterion,
                                                batch_size, device)
    train_acc, train_conf = print_acc_conf(infer_train_conf, train_label)

    logging.info("all test set")
    all_test_loss, infer_all_test_conf = undefendtest(all_test_data_tensor, all_test_label_tensor, net, criterion,
                                                      batch_size, device)
    all_test_acc, all_test_conf = print_acc_conf(infer_all_test_conf, all_test_label)

    logging.info("training tr set")
    tr_loss, infer_train_conf_tr = undefendtest(train_data_tr_attack_tensor, train_label_tr_attack_tensor, net,
                                                criterion, batch_size, device)
    tr_acc, tr_conf = print_acc_conf(infer_train_conf_tr, train_label_tr_attack)
    logging.info("training te set")
    te_loss, infer_train_conf_te = undefendtest(train_data_te_attack_tensor, train_label_te_attack_tensor, net,
                                                criterion, batch_size, device)
    te_acc, te_conf = print_acc_conf(infer_train_conf_te, train_label_te_attack)
    logging.info("test set")
    test_loss, infer_test_conf = undefendtest(test_data_tensor, test_label_tensor, net, criterion, batch_size, device)
    test_acc, test_conf = print_acc_conf(infer_test_conf, test_label)
    logging.info("reference set")
    ref_loss, infer_ref_conf = undefendtest(ref_data_tensor, ref_label_tensor, net, criterion, batch_size, device)
    ref_acc, ref_conf = print_acc_conf(infer_ref_conf, ref_label)

    logging.info("For comparison on undefend model")
    logging.info("avg acc  on train/all test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/"
                 "{:.4f}/{:.4f}/{:.4f}".format(train_acc, all_test_acc, tr_acc, te_acc, test_acc, ref_acc))
    logging.info("avg conf on train/all_test/tr/te/test/reference set: {:.4f}/{:.4f}/{:.4f}/"
                 "{:.4f}/{:.4f}/{:.4f}".format(train_conf, all_test_conf, tr_conf, te_conf, test_conf, ref_conf))
    system_attack(infer_train_conf_tr, train_label_tr_attack, infer_train_conf_te, train_label_te_attack,
                  infer_ref_conf, ref_label, infer_test_conf, test_label, device, num_class=num_class,
                  attack_epochs=epochs, batch_size=batch_size)


def save_checkpoint(state, checkpoint, filename='checkpoint.pth.tar'):
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def train(train_data, train_labels, model, criterion, optimizer, batch_size, device):
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

    return losses.avg, top1.avg


def train_eval(test_data, labels, model, criterion, batch_size, device):
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

    return losses.avg, top1.avg, confs.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def undefendtest(test_data, labels, model, criterion, batch_size, device):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    len_t = int(np.ceil(len(test_data) / batch_size))
    infer_np = np.zeros((len(test_data), 100))

    for batch_ind in range(len_t):
        end_idx = min(len(test_data), (batch_ind + 1) * batch_size)
        inputs = test_data[batch_ind * batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind * batch_size: end_idx].to(device, torch.long)

        # compute output
        outputs = model(inputs)
        infer_np[batch_ind * batch_size: end_idx] = (F.softmax(outputs, dim=1)).detach().cpu().numpy()

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size()[0])

    return losses.avg, infer_np


def print_acc_conf(infer_np, test_labels):
    cal = np.zeros(19)
    calacc = np.zeros(19)
    conf_metric = np.max(infer_np, axis = 1)
    conf_metric_ind = np.argmax(infer_np, axis = 1)
    conf_avg = np.mean(conf_metric)
    acc_avg = np.mean(conf_metric_ind==test_labels)
    logging.info("Total data: {:d}. Average acc: {:.4f}. Average confidence: {:.4f}.".format(len(infer_np)
                                                                                             , acc_avg, conf_avg))

    return acc_avg, conf_avg


def system_attack(train_member_pred, train_member_label, test_member_pred, test_member_label, train_nonmember_pred,
                  train_nonmember_label, test_nonmember_pred, test_nonmember_label, device, num_class=100,
                  attack_epochs=150, batch_size=512):
    len1, len2 = min(len(train_member_pred), len(train_nonmember_pred)), min(len(test_member_pred),
                                                                             len(test_nonmember_pred))

    train_member_pred, train_member_label = train_member_pred[:len1], train_member_label[:len1]
    train_nonmember_pred, train_nonmember_label = train_nonmember_pred[:len1], train_nonmember_label[:len1]
    test_member_pred, test_member_label = test_member_pred[:len2], test_member_label[:len2]
    test_nonmember_pred, test_nonmember_label = test_nonmember_pred[:len2], test_nonmember_label[:len2]

    logging.info("Evaluating direct single-query attacks : {}, {}, {}, {}".format(len(train_member_pred),
                 len(train_nonmember_pred), len(test_member_pred), len(test_nonmember_pred)))
    logging.info("batch_size: {}".format(batch_size))
    logging.info(train_member_label[:20])
    logging.info(test_member_label[:20])
    logging.info(test_nonmember_label[:20])
    logging.info(train_nonmember_label[:20])
    logging.info(
        'classifier acc on attack training set: {:.4f}, {:.4f}.\n'
        'classifier acc on attack test set: {:.4f}, {:.4f}.'.format(
            np.mean(np.argmax(train_member_pred, axis=1) == train_member_label),
            np.mean(np.argmax(train_nonmember_pred, axis=1) == train_nonmember_label),
            np.mean(np.argmax(test_member_pred, axis=1) == test_member_label),
            np.mean(np.argmax(test_nonmember_pred, axis=1) == test_nonmember_label)))

    train_mem_stat = get_conf(train_member_pred, train_member_label)
    train_nonmem_stat = get_conf(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = get_conf(test_member_pred, test_member_label)
    test_nonmem_stat = get_conf(test_nonmember_pred, test_nonmember_label)
    conf_acc, conf_acc_g, _, conf_acc_c, _ = threshold_based_inference_attack(train_mem_stat, train_member_label,
                                                                              train_nonmem_stat, train_nonmember_label,
                                                                              test_mem_stat, test_member_label,
                                                                              test_nonmem_stat, test_nonmember_label)

    train_mem_stat = -get_entropy(train_member_pred)
    train_nonmem_stat = -get_entropy(train_nonmember_pred)
    test_mem_stat = -get_entropy(test_member_pred)
    test_nonmem_stat = -get_entropy(test_nonmember_pred)
    entr_acc, entr_acc_g, _, entr_acc_c, _ = threshold_based_inference_attack(train_mem_stat, train_member_label,
                                                                              train_nonmem_stat, train_nonmember_label,
                                                                              test_mem_stat, test_member_label,
                                                                              test_nonmem_stat, test_nonmember_label)

    train_mem_stat = -get_mentropy(train_member_pred, train_member_label)
    train_nonmem_stat = -get_mentropy(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = -get_mentropy(test_member_pred, test_member_label)
    test_nonmem_stat = -get_mentropy(test_nonmember_pred, test_nonmember_label)
    mentr_acc, mentr_acc_g, _, mentr_acc_c, _ = threshold_based_inference_attack(train_mem_stat, train_member_label,
                                                                                 train_nonmem_stat,
                                                                                 train_nonmember_label, test_mem_stat,
                                                                                 test_member_label, test_nonmem_stat,
                                                                                 test_nonmember_label)

    train_mem_stat = get_correct(train_member_pred, train_member_label)
    train_nonmem_stat = get_correct(train_nonmember_pred, train_nonmember_label)
    test_mem_stat = get_correct(test_member_pred, test_member_label)
    test_nonmem_stat = get_correct(test_nonmember_pred, test_nonmember_label)
    corr_acc, _ = threshold_based_inference_attack(train_mem_stat, train_member_label, train_nonmem_stat,
                                                   train_nonmember_label, test_mem_stat, test_member_label,
                                                   test_nonmem_stat, test_nonmember_label, per_class=False)

    nn_acc, _ = nn_attack(train_member_pred, train_member_label, train_nonmember_pred, train_nonmember_label,
                          test_member_pred, test_member_label, test_nonmember_pred, test_nonmember_label, device,
                          attack_epochs=150, batch_size=512, num_class=100)

    logging.info(
        "Best direct single-query attack acc: {:.4f}. NN attack: {:.4f}. Correctness: {:.4f}. Global|Class:  "
        "Conf:{:.4f}|{:.4f}. Entr: {:.4f}|{:.4f}. Mentr: {:.4f}|{:.4f}".format(
            max(entr_acc, mentr_acc, conf_acc, corr_acc, nn_acc), nn_acc, corr_acc, conf_acc_g, conf_acc_c, entr_acc_g,
            entr_acc_c, mentr_acc_g, mentr_acc_c))

    return max(entr_acc, mentr_acc, conf_acc, corr_acc, nn_acc)


def train_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, attack_optimizer,
                 batch_size, device, train_mode=0):
    # switch to train mode
    attack_model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data / batch_size))
    for batch_ind in range(len_t):
        end_idx = min((batch_ind + 1) * batch_size, len_data)

        outputs = infer_data[batch_ind * batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind * batch_size: end_idx].to(device, torch.long)
        outputs_non = attack_infer_data[batch_ind * batch_size: end_idx].to(device, torch.float)
        targets_attack = attack_labels[batch_ind * batch_size: end_idx].to(device, torch.long)

        comb_inputs = torch.cat((outputs, outputs_non))
        comb_targets = torch.cat((targets, targets_attack)).view([-1, 1]).to(device, torch.float)

        if train_mode != 2:
            one_hot_tr = torch.zeros(comb_inputs.size()[0], comb_inputs.size()[1]).to(device, torch.float)
            target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).data, 1)
            attack_output = attack_model(comb_inputs, target_one_hot).view([-1])
        else:
            attack_output = attack_model(comb_inputs).view([-1])

        att_labels = torch.zeros((outputs.shape[0] + outputs_non.shape[0]))
        att_labels[:outputs.shape[0]] = 1.0
        att_labels[outputs.shape[0]:] = 0.0
        is_member_labels = att_labels.to(device, torch.float)

        loss_attack = attack_criterion(attack_output, is_member_labels)

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy() > 0.5), (is_member_labels.data.cpu().numpy() > 0.5)))

        losses.update(loss_attack.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])

        # compute gradient and do SGD step
        attack_optimizer.zero_grad()
        loss_attack.backward()
        attack_optimizer.step()

    return losses.avg, top1.avg


def test_attack(infer_data, labels, attack_infer_data, attack_labels, attack_model, attack_criterion, batch_size,
                device, train_mode=0):
    attack_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    len_data = min(len(labels), len(attack_labels))
    len_t = int(np.ceil(len_data / batch_size))
    member_prob = np.zeros(len_data)
    nonmember_prob = np.zeros(len_data)

    for batch_ind in range(len_t):
        end_idx = min(len_data, (batch_ind + 1) * batch_size)

        outputs = infer_data[batch_ind * batch_size: end_idx].to(device, torch.float)
        targets = labels[batch_ind * batch_size: end_idx].to(device, torch.long)
        outputs_non = attack_infer_data[batch_ind * batch_size: end_idx].to(device, torch.float)
        targets_attack = attack_labels[batch_ind * batch_size: end_idx].to(device, torch.long)

        comb_inputs = torch.cat((outputs, outputs_non))
        comb_targets = torch.cat((targets, targets_attack)).view([-1, 1]).to(device, torch.float)

        if train_mode != 2:
            one_hot_tr = torch.zeros(comb_inputs.size()[0], comb_inputs.size()[1]).to(device, torch.float)
            target_one_hot = one_hot_tr.scatter_(1, comb_targets.to(device, torch.long).view([-1, 1]).data, 1)

            attack_output = attack_model(comb_inputs, target_one_hot).view([-1])
        else:
            attack_output = attack_model(comb_inputs).view([-1])

        att_labels = torch.zeros((outputs.shape[0] + outputs_non.size()[0]))
        att_labels[:outputs.shape[0]] = 1.0
        att_labels[outputs.shape[0]:] = 0.0

        is_member_labels = att_labels.to(device, torch.float)

        loss = attack_criterion(attack_output, is_member_labels)

        member_prob[batch_ind * batch_size: end_idx] = attack_output.data.cpu().numpy()[: outputs.shape[0]]
        nonmember_prob[batch_ind * batch_size: end_idx] = attack_output.data.cpu().numpy()[outputs.shape[0]:]

        prec1 = np.mean(np.equal((attack_output.data.cpu().numpy() > 0.5), (is_member_labels.data.cpu().numpy() > 0.5)))
        losses.update(loss.item(), comb_inputs.size()[0])
        top1.update(prec1, comb_inputs.size()[0])

    return losses.avg, top1.avg, member_prob, nonmember_prob


def get_entropy(sample_predictions):
    # calculte entropy given prediction vectors(N,C) and labels(N), where N is the size of samples
    # and C is the number of class.
    # lower is likely to be samples
    outputs = sample_predictions.copy()
    outputs[outputs <= 0] = small_delta
    return np.sum(-outputs * np.log(outputs), axis=1)


def get_mentropy(sample_predictions, sample_labels):
    # calculate modified entropy given prediction vectors(N,C) and labels(N), where N is the size of samples
    # and C is the number of class.
    # lower is likely to be samples
    outputs = sample_predictions.copy()
    outputs[np.arange(len(sample_predictions)), sample_labels] = 1 - outputs[
        np.arange(len(sample_predictions)), sample_labels]
    outputs2 = 1 - outputs
    outputs2[outputs == 0] = small_delta
    return np.sum(-outputs * np.log(outputs2), axis=1)


def get_conf(sample_predictions, sample_labels):
    # higher is likely to be samples
    return sample_predictions[np.arange(len(sample_predictions)), sample_labels]


def get_correct(sample_predictions, sample_labels):
    # higher is likely to be samples
    return (np.argmax(sample_predictions, axis=1) == sample_labels).astype(int)


def threshold_based_inference_attack(train_member_stat, train_member_label, train_nonmember_stat, train_nonmember_label,
                                     test_member_stat, test_member_label, test_nonmember_stat, test_nonmember_label,
                                     num_class=100, per_class=True):
    """
    train_member_stat: member samples for finding threshold
    train_nonmember_stat: nonmember samples for finding threshold
    test_member_stat: member samples for MIA
    test_nonmember_stat: nonmember samples for evaluation MIA
    Note: Both stats are assumed to behave like confidence values, i.e., higher is better. Negate the values if it
    behaves in the opposite way, e.g., for xe-loss, lower is better
    """
    # global threshold
    list_all = np.concatenate((train_member_stat, train_nonmember_stat))
    max_gap = 0
    thre_chosen_g = 0
    list_all.sort()
    for thre in list_all:
        ratio1 = np.sum(train_member_stat >= thre)
        ratio2 = len(train_nonmember_stat) - np.sum(train_nonmember_stat >= thre)
        if ratio1 + ratio2 > max_gap:
            max_gap = ratio1 + ratio2
            thre_chosen_g = thre
    # evaluate global threshold
    ratio1 = np.sum(test_member_stat >= thre_chosen_g)
    ratio2 = len(test_nonmember_stat) - np.sum(test_nonmember_stat >= thre_chosen_g)
    global_MIA_acc = (ratio1 + ratio2) / (len(test_member_stat) + len(test_nonmember_stat))

    if per_class == True:
        # per-class threshold
        thre_chosen_class = np.zeros(num_class)
        for i in range(num_class):
            train_member_stat_class = train_member_stat[train_member_label == i]
            train_nonmember_stat_class = train_nonmember_stat[train_nonmember_label == i]
            list_all_class = np.concatenate((train_member_stat_class, train_nonmember_stat_class))
            max_gap = 0
            thre_chosen = 0
            list_all_class.sort()
            for thre in list_all_class:
                ratio1 = np.sum(train_member_stat_class >= thre)
                ratio2 = len(train_nonmember_stat_class) - np.sum(train_nonmember_stat_class >= thre)
                if ratio1 + ratio2 > max_gap:
                    max_gap = ratio1 + ratio2
                    thre_chosen = thre
            thre_chosen_class[i] = thre_chosen
        # evaluate per class threshold
        ratio1 = np.sum(test_member_stat >= thre_chosen_class[test_member_label])
        ratio2 = len(test_nonmember_stat) - np.sum(test_nonmember_stat >= thre_chosen_class[test_nonmember_label])
        class_MIA_acc = (ratio1 + ratio2) / (len(test_member_stat) + len(test_nonmember_stat))
        return max(global_MIA_acc, class_MIA_acc), global_MIA_acc, thre_chosen_g, class_MIA_acc, thre_chosen_class
    else:
        return global_MIA_acc, thre_chosen_g


def nn_attack(train_member_pred, train_member_label, train_nonmember_pred, train_nonmember_label, test_member_pred,
              test_member_label, test_nonmember_pred, test_nonmember_label, device, attack_epochs=150, batch_size=512,
              num_class=100, train_mode=0):
    """
    This assumes len(train_member_pred)==len(tran_nonmember_pred) and len(test_member_pred)==len(test_nonmember_pred)
    """
    test_member_pred_tensor = torch.from_numpy(test_member_pred).type(torch.FloatTensor)
    test_member_label_tensor = torch.from_numpy(test_member_label).type(torch.LongTensor)
    test_nonmember_pred_tensor = torch.from_numpy(test_nonmember_pred).type(torch.FloatTensor)
    test_nonmember_label_tensor = torch.from_numpy(test_nonmember_label).type(torch.LongTensor)

    if train_mode != 2:
        attack_model = InferenceAttack_HZ(num_class).to(device, torch.float)
        attack_criterion = nn.MSELoss().to(device, torch.float)
        attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
    else:
        attack_model = InferenceAttack_HZWL(num_class).to(device, torch.float)
        attack_criterion = nn.MSELoss().to(device, torch.float)
        attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)

    best_nn_acc = 0.0

    for epoch in range(0, attack_epochs):
        r = np.arange(len(train_member_pred))
        np.random.shuffle(r)
        train_member_pred = train_member_pred[r]
        train_member_label = train_member_label[r]
        r = np.arange(len(train_nonmember_pred))
        train_nonmember_pred = train_nonmember_pred[r]
        train_nonmember_label = train_nonmember_label[r]

        train_member_pred_tensor = torch.from_numpy(train_member_pred).type(torch.FloatTensor)
        train_member_label_tensor = torch.from_numpy(train_member_label).type(torch.LongTensor)
        train_nonmember_pred_tensor = torch.from_numpy(train_nonmember_pred).type(torch.FloatTensor)
        train_nonmember_label_tensor = torch.from_numpy(train_nonmember_label).type(torch.LongTensor)

        train_loss, train_attack_acc = train_attack(train_member_pred_tensor, train_member_label_tensor,
                                                    train_nonmember_pred_tensor, train_nonmember_label_tensor,
                                                    attack_model, attack_criterion, attack_optimizer, batch_size,
                                                    train_mode)
        test_loss, test_attack_acc, mem, nonmem = test_attack(test_member_pred_tensor, test_member_label_tensor,
                                                              test_nonmember_pred_tensor, test_nonmember_label_tensor,
                                                              attack_model, attack_criterion, batch_size, train_mode)

        is_best = test_attack_acc > best_nn_acc
        best_nn_acc = max(test_attack_acc, best_nn_acc)

    return best_nn_acc, attack_model
