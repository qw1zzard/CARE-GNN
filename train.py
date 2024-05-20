import argparse
import os
import random
import time

from sklearn.model_selection import train_test_split

import wandb
from graphsage import *  # noqa: F403
from layers import *  # noqa: F403
from model import *  # noqa: F403
from utils import *  # noqa: F403

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""
	Training CARE-GNN
	Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
	Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# dataset and model dependent args
parser.add_argument(
    '--data', type=str, default='yelp', help='The dataset name. [yelp, amazon]'
)
parser.add_argument(
    '--model', type=str, default='CARE', help='The model name. [CARE, SAGE]'
)
parser.add_argument(
    '--inter',
    type=str,
    default='GNN',
    help='The inter-relation aggregator type. [Att, Weight, Mean, GNN]',
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=1024,
    help='Batch size 1024 for yelp, 256 for amazon.',
)

# hyper-parameters
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda_1', type=float, default=2, help='Simi loss weight.')
parser.add_argument(
    '--lambda_2', type=float, default=1e-3, help='Weight decay (L2 loss weight).'
)
parser.add_argument(
    '--emb-size', type=int, default=64, help='Node embedding size at the last layer.'
)
parser.add_argument('--num-epochs', type=int, default=31, help='Number of epochs.')
parser.add_argument(
    '--test-epochs', type=int, default=3, help='Epoch interval to run test set.'
)
parser.add_argument('--under-sample', type=int, default=1, help='Under-sampling scale.')
parser.add_argument('--step-size', type=float, default=2e-2, help='RL action step size')

# other args
parser.add_argument(
    '--no-cuda', action='store_true', default=False, help='Disables CUDA training.'
)
parser.add_argument('--seed', type=int, default=72, help='Random seed.')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  # noqa: F405
print(f'run on {args.data}')


wandb.init(
    # set the wandb project where this run will be logged
    project='care-gnn',
    # track hyperparameters and run metadata
    config={
        'data': args.data,
        'model': args.model,
        'inter': args.inter,
        'batch-size': args.batch_size,
        'lr': args.lr,
        'lambda_1': args.lambda_1,
        'lambda_2': args.lambda_2,
        'emb-size': args.emb_size,
        'num-epochs': args.num_epochs,
        'test-epochs': args.test_epochs,
        'under-sample': args.under_sample,
        'step-size': args.step_size,
        'no-cuda': args.no_cuda,
        'seed': args.seed,
    },
)


# load graph, feature, and label
[homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data)  # noqa: F405

# train_test split
np.random.seed(args.seed)  # noqa: F405
random.seed(args.seed)
if args.data == 'yelp':
    index = list(range(len(labels)))
    idx_train, idx_test, y_train, y_test = train_test_split(
        index, labels, stratify=labels, test_size=0.60, random_state=2, shuffle=True
    )
elif args.data == 'amazon':  # amazon
    # 0-3304 are unlabeled nodes
    index = list(range(3305, len(labels)))
    idx_train, idx_test, y_train, y_test = train_test_split(
        index,
        labels[3305:],
        stratify=labels[3305:],
        test_size=0.60,
        random_state=2,
        shuffle=True,
    )

# split pos neg sets for under-sampling
train_pos, train_neg = pos_neg_split(idx_train, y_train)  # noqa: F405

# initialize model input
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])  # noqa: F405
feat_data = normalize(feat_data)  # noqa: F405
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)  # noqa: F405
if args.cuda:
    features.cuda()

# set input graph
if args.model == 'SAGE':
    adj_lists = homo
else:
    adj_lists = [relation1, relation2, relation3]

print(f'Model: {args.model}, Inter-AGG: {args.inter}, emb_size: {args.emb_size}.')

# build one-layer models
if args.model == 'CARE':
    intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)  # noqa: F405
    intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)  # noqa: F405
    intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)  # noqa: F405
    inter1 = InterAgg(  # noqa: F405
        features,
        feat_data.shape[1],
        args.emb_size,
        adj_lists,
        [intra1, intra2, intra3],
        inter=args.inter,
        step_size=args.step_size,
        cuda=args.cuda,
    )
elif args.model == 'SAGE':
    agg1 = MeanAggregator(features, cuda=args.cuda)  # noqa: F405
    enc1 = Encoder(  # noqa: F405
        features,
        feat_data.shape[1],
        args.emb_size,
        adj_lists,
        agg1,
        gcn=True,
        cuda=args.cuda,
    )

if args.model == 'CARE':
    gnn_model = OneLayerCARE(2, inter1, args.lambda_1)  # noqa: F405
elif args.model == 'SAGE':
    # the vanilla GraphSAGE model as baseline
    enc1.num_samples = 5
    gnn_model = GraphSage(2, enc1)  # noqa: F405

if args.cuda:
    gnn_model.cuda()

optimizer = torch.optim.Adam(  # noqa: F405
    filter(lambda p: p.requires_grad, gnn_model.parameters()),
    lr=args.lr,
    weight_decay=args.lambda_2,
)
times = []
performance_log = []

# train the model
for epoch in range(args.num_epochs):
    # randomly under-sampling negative nodes for each epoch
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)  # noqa: F405
    rd.shuffle(sampled_idx_train)  # noqa: F405

    # send number of batches to model to let the RLModule know the training progress
    num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
    if args.model == 'CARE':
        inter1.batch_num = num_batches

    loss = 0.0
    epoch_time = 0

    # mini-batch training
    for batch in range(num_batches):
        start_time = time.time()
        i_start = batch * args.batch_size
        i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = labels[np.array(batch_nodes)]  # noqa: F405
        optimizer.zero_grad()
        if args.cuda:
            loss = gnn_model.loss(
                batch_nodes,
                Variable(torch.tensor(batch_label, dtype=torch.long, device='cuda')),  # noqa: F405
            )
        else:
            loss = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))  # noqa: F405
        loss.backward()
        optimizer.step()
        end_time = time.time()
        epoch_time += end_time - start_time
        loss += loss.item()

    loss_score = float(loss.item() / num_batches)
    print(f'Epoch: {epoch}, loss: {loss_score}, time: {epoch_time}s')
    wandb.log({'epoch': epoch, 'loss': loss_score, 'time': epoch_time})

    # testing the model for every $test_epoch$ epoch
    if epoch % args.test_epochs == 0:
        if args.model == 'SAGE':
            test_sage(idx_test, y_test, gnn_model, args.batch_size)  # noqa: F405
        else:
            gnn_auc, f1, label_auc, gnn_recall, label_recall = test_care(  # noqa: F405
                idx_test, y_test, gnn_model, args.batch_size
            )
            performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])
            wandb.log({'auc': gnn_auc, 'f1': f1})
