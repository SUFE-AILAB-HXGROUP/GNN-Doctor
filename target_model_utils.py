import logging

import torch

from gnns import GCN, UnsupGCN


def target_model_train(dataset, setting, attack, target_model, epochs, device):
    """
    :param dataset: name of graph dataset
    :param setting: attack setting - black_box or node_emb_publish
    :param target_model: name of target_model
    :return: trained model
    """
    if setting == 'node_emb_publish':
        if target_model == 'gcn':
            train_data, val_data, test_data = dataset[0]
            train_data.to(device)
            val_data.to(device)
            test_data.to(device)
            model = UnsupGCN(input_dim=dataset.num_features, hidden_dim=128, output_dim=64).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            best_val_auc = final_test_auc = 0
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()
                loss = model.compute_loss(train_data)
                loss.backward()
                optimizer.step()

                model.eval()
                val_auc = model.test_auc(val_data)
                test_auc = model.test_auc(test_data)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    final_test_auc = test_auc
                logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

            logging.info(f'Final Test: {final_test_auc:.4f}')

        return model

    elif setting == 'black_box':
        if attack == 'membership_infer':
            return None
        else:
            if target_model == 'gcn':
                data = dataset[0]
                data.to(device)
                num_features = data.x.size(1)
                num_classes = len(set(data.y.cpu().numpy()))
                model = GCN(input_dim=num_features, hidden_dim=128, output_dim=num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                best_val_acc = test_acc = 0
                for epoch in range(1, epochs + 1):
                    model.train()
                    optimizer.zero_grad()
                    y_pred = model(data)
                    loss = model.compute_loss(y_pred[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    logits = model(data)
                    accs = []
                    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                        acc = model.test_model(logits, data.y, mask)
                        accs.append(acc)
                    train_acc, val_acc, tmp_test_acc = accs[0], accs[1], accs[2]
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        test_acc = tmp_test_acc
                    logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                                 f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

            return model

    else:
        logging.error("Not implemented setting. Please set setting=='node_emb_publish' or 'black_box'.")

        return None
