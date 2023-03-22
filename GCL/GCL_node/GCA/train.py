import random
from torch_geometric.utils import dropout_adj
from .model import Encoder, GRACE
from .functional import *
from .eval import *
from .utils import *
from ..dataset import get_dataset


def drop_edge(data, args, drop_weights, drop_edge_rate):
    if args.drop_scheme == 'uniform':
        return dropout_adj(data.edge_index, p=drop_edge_rate)[0]
    elif args.drop_scheme in ['degree', 'evc', 'pr']:
        return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate, threshold=0.7)
    else:
        raise Exception(f'undefined drop scheme: {args.drop_scheme}')


def train(model, optimizer, data, args, feature_weights, drop_weights):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = drop_edge(data, args, drop_weights, args.drop_edge_rate_1)
    edge_index_2 = drop_edge(data, args, drop_weights, args.drop_edge_rate_2)
    x_1 = drop_feature(data.x, args.drop_feature_rate_1)
    x_2 = drop_feature(data.x, args.drop_feature_rate_2)

    if args.drop_scheme in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, args.drop_feature_rate_1)
        x_2 = drop_feature_weighted_2(data.x, feature_weights, args.drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, dataset, args, split):
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
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    return acc


def GCA(args, logger):
    args.learning_rate = 0.01
    args.num_hidden = 256
    args.num_proj_hidden = 32
    args.num_layers = 2
    args.activation = 'prelu'
    args.base_model = 'GCNConv'
    args.drop_edge_rate_1 = 0.3
    args.drop_edge_rate_2 = 0.4
    args.drop_feature_rate_1 = 0.1
    args.drop_feature_rate_2 = 0.0
    args.tau = 0.4
    args.num_epochs = 1000
    args.weight_decay = 1e-5

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = get_dataset(args.path, args.dataset)

    data = dataset[0]
    device = args.device
    data = data.to(device)

    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    encoder = Encoder(dataset.num_features, args.num_hidden, get_activation(args.activation),
                      base_model=get_base_model(args.base_model), k=args.num_layers).to(device)
    model = GRACE(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif args.drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif args.drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if args.drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif args.drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif args.drop_scheme == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    for epoch in range(1, args.num_epochs + 1):
        loss = train(model, optimizer, data, args, feature_weights, drop_weights)
        logger.info(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            acc = test(model, data, dataset, args, split)
            logger.info(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    acc = test(model, data, dataset, args, split)
    logger.info(f'{acc}')
