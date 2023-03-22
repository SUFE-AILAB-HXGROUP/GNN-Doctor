from .link_steal import *
from .link_infer import *
from .attack_models import *

from torch_geometric.utils import to_dense_adj


def attack_func(dataset, model, setting, attack_method, attack_epoch, device):
    """
    :param dataset: processed dataset
    :param model: trained target model
    :param setting: attack setting - black box or node emb publish
    :param attack_method: name of attack
    :param attack_epoch: epoch of training attack model
    :return:
    """
    if setting == 'node_emb_publish':
        train_data, val_data, test_data = dataset[0]
        train_data.to(device)
        val_data.to(device)
        test_data.to(device)
        if attack_method == 'link_infer':
            attack_edges = test_data.edge_label_index.cpu().numpy().T
            attack_edge_labels = test_data.edge_label.cpu().numpy()
            model.eval()
            embs = model.get_emb(train_data.x, train_data.edge_index)
            embs = embs.cpu().numpy()
            metric_list, attack_auc_list = sim_attacks(embs, attack_edges, attack_edge_labels)

            shadow_edges = val_data.edge_label_index.cpu().numpy()
            shadow_edge_labels = val_data.edge_label.cpu().numpy()
            ml_attacks(embs, shadow_edges, shadow_edge_labels, attack_edges.T, attack_edge_labels)

        elif attack_method == 'attr_infer':
            model.eval()
            embs = model.get_emb(train_data.x, train_data.edge_index)
            evaluator = MulticlassEvaluator()
            acc = log_regression(embs, train_data, evaluator, split='rand:0.1',
                                 num_epochs=attack_epoch, test_device=device)['acc']
            logging.info("Node attr inference Acc: {:.4f}".format(acc))

        elif attack_method == 'membership_infer':
            model.eval()
            embs = model.get_emb(train_data.x, train_data.edge_index)
            embs = embs.cpu().numpy()
            logging.info("To be implemented.")

        elif attack_method == 'model_inver':
            logging.info("To be implemented.")

    elif setting == 'black_box':
        if attack_method == 'link_steal':
            data = dataset[0]
            data.to(device)
            model.eval()
            y_pred = model(data).cpu().detach().numpy().tolist()

            adj = to_dense_adj(data.edge_index.cpu())[0].numpy()
            node_num = adj.shape[0]
            link, unlink = get_link(adj, node_num)

            random.shuffle(link)
            random.shuffle(unlink)
            edge_label = []
            for row in link:
                edge_label.append(1)
            for row in unlink:
                edge_label.append(0)

            train_ratio = 1 / 10.0
            generate_train_test(link, unlink, y_pred, train_ratio)

            test_path = "link_steal_tmp/train_ratio_%s_test.json" % str(train_ratio)

            test_data = open(test_path).readlines()  # read test data only
            label_list = []
            target_posterior_list = []

            for row in test_data:
                row = json.loads(row)
                label_list.append(row["label"])
                target_posterior_list.append([row["y_pred0"], row["y_pred1"]])

            sim_list_target = attack_0(target_posterior_list)
            write_auc(sim_list_target, label_list, desc="target posterior similarity")

        elif attack_method == 'attr_infer':
            logging.info("To be implemented.")
        elif attack_method == 'property_infer':
            logging.info("To be implemented")

    else:
        logging.error("Not implemented setting. Please set setting=='node_emb_publish' or 'black_box'.")