import random
import time
import json

import numpy as np
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, \
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.metrics import roc_auc_score


def get_link(adj, node_num):
    unlink = []
    link = []
    existing_set = set([])
    rows, cols = adj.nonzero()
    print("There are %d edges in this dataset" % len(rows))
    for i in range(len(rows)):
        r_index = rows[i]
        c_index = cols[i]
        if r_index < c_index:
            link.append([r_index, c_index])
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    t_start = time.time()
    while len(unlink) < len(link):
        if len(unlink) % 1000 == 0:
            print(len(unlink), time.time() - t_start)

        row = random.randint(0, node_num - 1)
        col = random.randint(0, node_num - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            existing_set.add(edge_str)
    return link, unlink


def generate_train_test(link, unlink, y_pred, train_ratio):
    train = []
    test = []

    train_len = len(link) * train_ratio
    for i in range(len(link)):
        link_id0 = link[i][0]
        link_id1 = link[i][1]

        line_link = {
            'label': 1,
            'y_pred0': y_pred[link_id0],
            'y_pred1': y_pred[link_id1],
            "id_pair":[int(link_id0),int(link_id1)]
        }

        unlink_id0 = unlink[i][0]
        unlink_id1 = unlink[i][1]

        line_unlink = {
            'label': 0,
            'y_pred0': y_pred[unlink_id0],
            'y_pred1': y_pred[unlink_id1],
            "id_pair":[int(unlink_id0),int(unlink_id1)]
        }

        if i < train_len:
            train.append(line_link)
            train.append(line_unlink)
        else:
            test.append(line_link)
            test.append(line_unlink)

    with open(
             "link_steal_tmp/train_ratio_%0.1f_train.json" %
              (train_ratio), "w") as wf1, open(
            "link_steal_tmp/train_ratio_%0.1f_test.json" %
            (train_ratio), "w") as wf2:
        for row in train:
            wf1.write("%s\n" % json.dumps(row))
        for row in test:
            wf2.write("%s\n" % json.dumps(row))
    return train, test


def attack_0(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target


def write_auc(pred_prob_list, label, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    with open("link_steal_tmp/attack_0.txt", "a") as wf:
        for i in range(len(sim_list_str)):
            pred = np.array(pred_prob_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            i_auc = roc_auc_score(label, pred)
            if i_auc < 0.5:
                i_auc = 1 - i_auc
            print(sim_list_str[i], i_auc)
            wf.write("%s,%d,%0.5f,%s\n" % ("attack0_%s_%s" % (desc, sim_list_str[i]), -1, i_auc, 0.5))