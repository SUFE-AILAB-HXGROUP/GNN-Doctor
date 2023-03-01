import copy
import numpy as np
import argparse
import os
import os.path as osp
import random
# import nni
import networkx as nx
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import torch
from torch_geometric import seed_everything
from torch_geometric.utils import dropout_adj, degree, to_undirected, to_networkx, \
    to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid, LastFMAsia, FacebookPagePage
import torch_geometric.transforms as T

from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, GRACE_priv
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense, drop_edge_weighted_combine
from pGRACE.eval import log_regression, MulticlassEvaluator, train_mlp
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality, random_split_graph
from pGRACE.dataset import get_dataset, get_cora_dataset

from priv_weights import shortest_path_weights, common_neighbor_weights, ori_graph_sp_masks, combine_weights, combine_weights_v2, dice_aug, edge_rand, lap_graph, get_priv_adj, get_weight_matrix
from utils import hadamard, l1_weight, l2_weight, concate, average, sim_attacks, write_auc


def train():
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            edges = drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            # t_delete = data.edge_index.shape[1] - edges.shape[1]
            # print('total delete: {}'.format(t_delete/2))
            # print(edges.shape)
            
            return edges
        elif param['drop_scheme'] in ['shortest_path', 'common_neighbor']:
            masks = torch.bernoulli(1-torch.Tensor(drop_weights)).to(torch.bool)
            return data.edge_index[:, masks]
        elif param['drop_scheme'] == 'combine':
            edges = drop_edge_weighted_combine(sp_masks, data.edge_index, de_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            # t_delete = data.edge_index.shape[1] - edges.shape[1]
            # sp_delete = sp_masks.sum().item()
            # other_delete = t_delete - sp_delete
            # print("total delete: {}, sp delete: {}, other delete: {}".format(t_delete/2, sp_delete/2, other_delete/2))
            # print(edges.shape)
            return edges
        
        elif param['drop_scheme'] == 'combine_v2':
            edges = drop_edge_weighted_combine(sp_masks, data.edge_index, de_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
            t_delete = data.edge_index.shape[1] - edges.shape[1]
            sp_delete = sp_masks.sum().item()
            other_delete = t_delete - sp_delete
            print("total delete: {}, sp delete: {}, other delete: {}".format(t_delete/2, sp_delete/2, other_delete/2))
            print(edges.shape)
            return edges
        elif param['drop_scheme'] == 'random':
            weights = [0.3]*data.edge_index.shape[1]
            masks = torch.bernoulli(1-torch.Tensor(weights)).to(torch.bool)
            return data.edge_index[:, masks]
        elif param['drop_scheme'] == 'dice':
            edges = dice_aug(data, test_data, p=param[f'drop_edge_rate_{idx}'])
            return edges
        elif param['drop_scheme'] == 'dp_edge_rand':
            sp_adj = to_scipy_sparse_matrix(data.edge_index)
            sp_adj = edge_rand(sp_adj, epsilon=10.2)
            edges = from_scipy_sparse_matrix(sp_adj)[0].to(data.edge_index.device)
            # print(data.edge_index.shape)
            # print(edges.shape)
            # print(edges.shape[1] - data.edge_index.shape[1])
            return edges
        
        elif param['drop_scheme'] == 'dp_lap_graph':
            sp_adj = to_scipy_sparse_matrix(data.edge_index)
            sp_adj = lap_graph(sp_adj, epsilon=0.15)
            edges = from_scipy_sparse_matrix(sp_adj)[0].to(data.edge_index.device)
            print(data.edge_index.shape)
            print(edges.shape)
            print(edges.shape[1] - data.edge_index.shape[1])
            
            return edges
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
    
    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    
    if param['drop_scheme'] in ['degree', 'pr',  'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    else:
        # x_1 = data.x
        # x_2 = data.x
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if args.feature == 'v1':
        x_1 = drop_feature_priv(x_1)
        x_2 = drop_feature_priv(x_2)
    elif args.feature == 'v2':
        x_1 = drop_feature_priv2(x_1)
        x_2 = drop_feature_priv2(x_2)

    # if args.feature:
    #     x_1 = drop_feature_priv(data.x)
    #     x_2 = drop_feature_priv(data.x)
    # else:
    #     x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    #     x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    # x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    # x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    # if param['drop_scheme'] in ['combine', 'combine_v2']:
    #     loss = model.loss_priv(z1, z2, weights_matrix, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    # else:
    #     loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)

    loss = model.loss_priv(z1, z2, weights_matrix, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(epoch, final=False):
    model.eval()

    z = model(data.x, data.edge_index)
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, data, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    # if final and use_nni:
    #     nni.report_final_result(acc)
    # elif use_nni:
    #     nni.report_intermediate_result(acc)
    
    return acc


# def link_eval():
#     model.eval()
#     # z = model(data.x, data.edge_index)
#     z = model(test_data.x, test_data.edge_index)
#     out = (z[test_data.edge_label_index[0]] * z[test_data.edge_label_index[1]]).sum(dim=-1)
#     out = out.view(-1).sigmoid()
#     auc = roc_auc_score(test_data.edge_label.cpu().numpy(), out.detach().cpu().numpy())
#     ap_score = average_precision_score(test_data.edge_label.cpu().numpy(), out.detach().cpu().numpy())
#
#     return auc, ap_score


def drop_feature_priv(x):
    drop_prob = torch.zeros_like(x)
    for i in range(test_edge_true2.shape[1]):
        drop_prob[test_edge_true2[random.choice([0, 1]), i],
                  x[test_edge_true2[0, i], :] == x[test_edge_true2[1, i], :]] = args.ratio
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[drop_mask] = 1 - x[drop_mask]
    return x


def drop_feature_priv2(x):
    drop_prob = torch.zeros_like(x)
    for i in range(test_edge_true2.shape[1]):
        drop_prob[test_edge_true2[random.choice([0, 1]), i],
                  x[test_edge_true2[0, i], :] == x[test_edge_true2[1, i], :]] = args.ratio
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[drop_mask] = 0.
    return x


def scenario_II_link_eval(epoch):
    model.eval()
    z = model(data.x, data.edge_index)
    train_1_mat = z[val_data.edge_label_index[0]].detach().cpu().numpy()
    train_2_mat = z[val_data.edge_label_index[1]].detach().cpu().numpy()
    test_1_mat = z[test_data.edge_label_index[0]].detach().cpu().numpy()
    test_2_mat = z[test_data.edge_label_index[1]].detach().cpu().numpy()
    train_y = val_data.edge_label.cpu().numpy()
    test_y = test_data.edge_label.cpu().numpy()
    # for cls_name in ['lr', 'xgb', 'mlp']:
    #     for op in ['hadamard', 'l1', 'l2', 'avg', 'concat']:
    for cls_name in ['xgb']:
        for op in ['hadamard']:
            if op == 'hadamard':
                train_x = hadamard(train_1_mat, train_2_mat)
                test_x = hadamard(test_1_mat, test_2_mat)
            elif op == 'l1':
                train_x = l1_weight(train_1_mat, train_2_mat)
                test_x = l1_weight(test_1_mat, test_2_mat)
            elif op == 'l2':
                train_x = l2_weight(train_1_mat, train_2_mat)
                test_x = l2_weight(test_1_mat, test_2_mat)
            elif op == 'avg':
                train_x = average(train_1_mat, train_2_mat)
                test_x = average(test_1_mat, test_2_mat)
            elif op == 'concat':
                train_x = concate(train_1_mat, train_2_mat)
                test_x = concate(test_1_mat, test_2_mat)
            
            if cls_name == 'lr':
                lgcls = LogisticRegression(random_state=42)
                lgcls.fit(train_x, train_y)
                y_pred = lgcls.predict(test_x)
            elif cls_name == 'xgb':
                xgbcls = xgb.XGBClassifier(random_state=42)
                xgbcls.fit(train_x, train_y)
                y_pred = xgbcls.predict(test_x)
            elif cls_name == 'mlp':
                y_pred = train_mlp(200, args.device, train_x, train_y, test_x, test_y)
            
            # f1 = f1_score(test_y, y_pred)
            auc = roc_auc_score(test_y, y_pred)
            # acc = accuracy_score(test_y, y_pred)
            ap_score = average_precision_score(test_y, y_pred)
            # print(cls_name, op, f1, auc, acc, ap_score)
            
            # with open('{}/{}_{}_{}_scenario2_results.txt'.format(
            #         final_folder, param['drop_scheme'], cls_name, op, args.ratio), 'a') as f:
            #     f.write("%d,%0.5f,%0.5f,%0.5f,%0.5f\n" % (epoch, f1, auc, acc, ap_score))
            #     f.flush()
                
            # with open('{}/scenario2_files/{}_{}_scenario2_results.txt'.format(final_folder, param['drop_scheme'], op), 'a') as f:
            # f.write('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(epoch, lg_acc, lg_auc, lg_ap_score, lg_f1, xgb_acc, xgb_auc, xgb_ap_score, xgb_f1))
            # f.flush()
    return ap_score, auc
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    # Cora, Citeseer, PubMed, lastfm_asia, facebookpagepage
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--param', type=str, default='local:cora.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--ratio', type=float, default=0.01)
    parser.add_argument('--weight', type=float, default=15)
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--drop_scheme', type=str, default='degree')  # degree, combine
    parser.add_argument('--feature', type=str, default='no')  # no, v1, v2
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1500,
        'weight_decay': 1e-5,
        # 'drop_scheme': 'combine',      # random, dice, degree, shortest_path, combine, combine_v2, dp_edge_rand, dp_lap_graph
        'adv': 'non'     # non, adv1, adv2
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    if args.dataset == 'facebookpagepage':
        args.param = 'local:fcpp.json'
    else:
        args.param = 'local:' + args.dataset.lower() + '.json'

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    param['drop_scheme'] = getattr(args, 'drop_scheme')
    # use_nni = args.param == 'nni'
    # if use_nni and args.device != 'cpu':
    #     args.device = 'cuda'

    torch_seed = args.seed
    torch.manual_seed(torch_seed)
    random.seed(12345)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load dataset
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
    ])
    
    # split graph
    if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'PubMed':
        dataset = Planetoid('datasets', args.dataset, transform=transform)
        ori_train_data, ori_val_data, ori_test_data = dataset[0]
    elif args.dataset == 'lastfm_asia':
        dataset = torch.load('datasets/lastfm_asia/processed/data.pt')[0]
        ori_train_data, ori_val_data, ori_test_data = transform(dataset)
        # dataset = LastFMAsia('datasets/lastfm_asia', args.dataset)
    elif args.dataset == 'facebookpagepage':
        dataset = torch.load('datasets/{}/processed/data.pt'.format(args.dataset))[0]
        ori_train_data, ori_val_data, ori_test_data = transform(dataset)
        # dataset = FacebookPagePage('datasets/facebookpagepage', args.dataset)
    else:
        raise NotImplementedError
    
    test_edges = ori_test_data.edge_label_index.numpy().T
    test_edge_labels = ori_test_data.edge_label.numpy()
    test_num = int(len(ori_test_data.edge_label) / 2)
    test_edge_true = ori_test_data.edge_label_index.numpy().T[:test_num, :]
    test_edge_false = ori_test_data.edge_label_index.numpy().T[test_num:, :]

    test_edge_true2 = ori_test_data.edge_label_index[:, :test_num]

    print('dataset {} loaded'.format(args.dataset))
    print('augmentation method: {}'.format(param['drop_scheme']))
    
    #########################################################
    
    topk = args.topk
    
    # 计算一个 private neighbors adjcency matrix
    num_nodes = ori_train_data.num_nodes
    priv_neighbor_adj = get_priv_adj(ori_test_data, num_nodes)
    
    # 计算一个权重矩阵
    # 1. 使用一个固定权重（先使用第一种）
    
    weights = args.weight
    weights_matrix = get_weight_matrix(priv_neighbor_adj, weights).to(device)
    
    # 2. 每对 private nodes 计算不同的权重，根据 shortest_path 计算，越短，weight 越大（最短是2）
    # weights_matrix = 
    
    #########################################################
    
    
    #########################################################
    
    final_folder = 'pagcl_{}_output_files'.format(args.dataset)
    if not osp.exists(final_folder):
        os.makedirs(final_folder)
        
    if not osp.exists(final_folder+'/scenario1_files'):
        os.makedirs(final_folder+'/scenario1_files')
    
    if not osp.exists(final_folder+'/scenario2_files'):
        os.makedirs(final_folder+'/scenario2_files')
        
    if not osp.exists(final_folder+'/tmp_emb_files'):
        os.makedirs(final_folder+'/tmp_emb_files')
    
    ##########################################################
    
    for iter_num in range(1, 2):
        torch_seed = iter_num
        torch.manual_seed(torch_seed)
        random.seed(torch_seed)
        
        data = copy.deepcopy(ori_train_data).to(device)
        test_data = copy.deepcopy(ori_test_data).to(device)
        val_data = copy.deepcopy(ori_val_data).to(device)
        
        # generate split
        split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
        if args.save_split:
            torch.save(split, args.save_split)
        elif args.load_split:
            split = torch.load(args.load_split)

        encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                          base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
        model = GRACE_priv(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=param['learning_rate'],
            weight_decay=param['weight_decay']
        )

        # edge drop weights
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        elif param['drop_scheme'] in ['shortest_path', 'common_neighbor']:
            if param['drop_scheme'] == 'shortest_path':
                # strategy - simple, threshold, betweenness
                drop_weights = shortest_path_weights(data, test_data, strategy='simple', rm_edge_num=3, initial_weight=0.3, sp_weight=0.9)
            elif param['drop_scheme'] == 'common_neighbor':
                drop_weights = common_neighbor_weights(data, test_data, initial_weight=0.3, cn_weight=0.9)
        elif param['drop_scheme'] == 'combine':
            # first get degree drop weights
            de_weights = degree_drop_weights(data.edge_index).to(device)
            sp_masks = combine_weights(de_weights, data, test_data, k=topk)
        elif param['drop_scheme'] == 'combine_v2':
            de_weights = degree_drop_weights(data.edge_index).to(device)
            sp_masks = combine_weights_v2(de_weights, data, test_data, k=3)
        else:
            drop_weights = None

        # feature drop weights
        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1], num_nodes=data.num_nodes)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        elif param['drop_scheme'] == 'pagcl':
            feature_weights = torch.ones((data.x.size(1),)).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        log = args.verbose.split(',')

        # with open('{}/scenario1_files/{}_results.txt'.format(final_folder, param['drop_scheme']), 'a') as file:
        #     file.write('pagcl method: {}, iter: {}\n'.format(param['drop_scheme'], iter_num))
        #     file.write('epoch\tnode classification acc\tlink prediction auc\n')
        #     file.flush()

        # for op in ['hadamard', 'l1', 'l2', 'avg', 'concat']:
        #     with open('{}/scenario2_files/{}_{}_scenario2_results.txt'.format(final_folder, param['drop_scheme'], op), 'a') as f:
        #         f.write('GCL method: {}, iter: {}, operation: {}\n'.format(param['drop_scheme'], iter_num, op))
        #         f.write('epoch\ts2 lg acc\ts2 lg auc\ts2 lg ap score\ts2 lg f1\ts2 xgb acc\ts2 xgb auc\ts2 xgb ap score\ts2 xgb f1\n')
        #         f.flush()
        ress = []
        accs = []
        for epoch in range(1, param['num_epochs'] + 1):
            if param['adv'] == 'non':
                loss = train()
                if 'train' in log:
                    print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

            # if epoch % 100 == 0:
                # save the model
            #     torch.save(model.state_dict(), '{}/tmp_emb_files/iter{}_epoch{}_{}_model.pt'.format(final_folder, iter_num, epoch, param['drop_scheme']))
            
            if epoch % 100 == 0:
                
                # save the model
                # torch.save(model.state_dict(), '{}/tmp_emb_files/iter{}_epoch{}_{}_model.pt'.format(final_folder, iter_num, epoch, param['drop_scheme']))
                
                model.eval()
                acc = test(epoch)
                z = model(test_data.x, test_data.edge_index).detach().cpu().numpy()
                # sim based attacks
                sim_list = sim_attacks(z, test_edges)
                
                ap1, auc1 = write_auc(args.dataset, epoch, "{}/scenario1_files".format(final_folder), sim_list, test_edge_labels)
                # link_auc = link_eval()
                # print("link eval auc: {}".format(link_auc))
                # machine learning based attacks
                ap2, auc2 = scenario_II_link_eval(epoch)
                
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}, link prediction auc = {auc2}')
                with open('{}/{}_{}_{}_{}_{}_results.txt'.format(
                        final_folder, param['drop_scheme'], args.weight, args.topk, args.ratio, args.feature),
                        'a') as file:
                    file.write(f'{args.weight}\t{epoch}\t{acc}\t{ap1}\t{auc1}\t{ap2}\t{auc2}\t\n')
                    file.flush()

                accs.append(acc)
                ress.append(f'\n{args.weight}\t{epoch}\t{acc}\t{ap1}\t{auc1}\t{ap2}\t{auc2}\t')

        # acc = test(epoch, final=True)
        #
        # if 'final' in log:
        #     print(f'{acc}')
        with open('{}/{}_{}_{}_{}_{}_results.txt'.format(
                final_folder, param['drop_scheme'], args.weight, args.topk, args.ratio, args.feature), 'a') as file:
            file.write(ress[np.argmax(accs)])
            file.flush()
