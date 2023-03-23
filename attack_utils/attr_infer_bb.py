from .attack_models import train_mlp

import logging

import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def attr_infer_bb_func(dataset, model, device, epochs):
    data = dataset[0]
    data = data.to(device)
    model.eval()
    embs = model.get_emb(data).detach().cpu().numpy()
    priv_label = data.priv_label.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(embs, priv_label, test_size=0.5, random_state=42)
    # train_mlp(epochs, device, X_train, y_train, X_test, y_test)
    for cls_name in ['lr', 'xgb']:
        if cls_name == 'lr':
            cls = LogisticRegression()
        else:
            cls = xgboost.XGBClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info("Attack Model:{} | Acc: {:.4f} | AUC: {:.4f} | F1: {:.4f}".format(cls_name, acc, auc, f1))

    # re-purpose attack
    # To be implemented