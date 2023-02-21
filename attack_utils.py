import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, pairwise_distances, f1_score, accuracy_score
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev, \
    braycurtis, canberra, cityblock, sqeuclidean
from sklearn.linear_model import LogisticRegression

from attack_models import train_mlp


def hadamard(x, y):
    return x * y


def l1_weight(x, y):
    return np.absolute(x - y)


def l2_weight(x, y):
    return np.square(x - y)


def concate(x, y):
    return np.concatenate((x, y), axis=1)


def average(x, y):
    return (x + y) / 2


def sim_attacks(embs, test_edges, label):
    # eight similarity metrics
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list = [[] for _ in range(len(sim_metric_list))]
    # record the similarities
    for edge in test_edges:
        for j in range(len(sim_metric_list)):
            sim = sim_metric_list[j](embs[edge[0]], embs[edge[1]])
            sim_list[j].append(sim)
    # compute auc score
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    auc_list = []
    for i in range(len(sim_list_str)):
        pred = np.array(sim_list[i], dtype=np.float64)
        where_are_nan = np.isnan(pred)
        where_are_inf = np.isinf(pred)
        pred[where_are_nan] = 0
        pred[where_are_inf] = 0

        i_auc = roc_auc_score(label, pred)
        if i_auc < 0.5:
            i_auc = 1 - i_auc
        print(sim_list_str[i], i_auc)
        auc_list.append(i_auc)

    return sim_list_str, auc_list


def ml_attacks(embs, train_edges, train_y, test_edges, test_y):
    train_mat_1 = embs[train_edges[0]]
    train_mat_2 = embs[train_edges[1]]
    test_mat_1 = embs[test_edges[0]]
    test_mat_2 = embs[test_edges[1]]

    for cls_name in ['lr', 'xgb', 'mlp']:
        for op in ['hadamard', 'l1', 'l2', 'avg', 'concat']:
            if op == 'hadamard':
                train_x = hadamard(train_mat_1, train_mat_2)
                test_x = hadamard(test_mat_1, test_mat_2)
            elif op == 'l1':
                train_x = l1_weight(train_mat_1, train_mat_2)
                test_x = l1_weight(test_mat_1, test_mat_2)
            elif op == 'l2':
                train_x = l2_weight(train_mat_1, train_mat_2)
                test_x = l2_weight(test_mat_1, test_mat_2)
            elif op == 'avg':
                train_x = average(train_mat_1, train_mat_2)
                test_x = average(test_mat_1, test_mat_2)
            elif op == 'concat':
                train_x = concate(train_mat_1, train_mat_2)
                test_x = concate(test_mat_1, test_mat_2)

            if cls_name == 'lr':
                lgcls = LogisticRegression(random_state=42)
                lgcls.fit(train_x, train_y)
                y_pred = lgcls.predict(test_x)
            elif cls_name == 'xgb':
                xgbcls = xgb.XGBClassifier(random_state=42)
                xgbcls.fit(train_x, train_y)
                y_pred = xgbcls.predict(test_x)
            elif cls_name == 'mlp':
                y_pred = train_mlp(200, 'cpu', train_x, train_y, test_x, test_y)

            f1 = f1_score(test_y, y_pred)
            auc = roc_auc_score(test_y, y_pred)
            acc = accuracy_score(test_y, y_pred)

            print(
                "cls: {}, operation: {}, f1 score: {:.4f}, auc score: {:.4f}, acc: {:.4f}".format(cls_name, op, f1, auc,
                                                                                                  acc))