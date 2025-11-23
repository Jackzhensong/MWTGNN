import warnings
import os, sys

import torch
from tqdm import trange
from datasets import DataLoader
from utils import *
from load_model import parse_model
import argparse
import time
from utils_wavelet import *

from config import seed_everything
import subprocess


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    return test_acc


@torch.no_grad()
def valid(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc



if __name__ == "__main__":
    # PARSER BLOCK
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-D', type=str, default='cora')
    parser.add_argument('--baseseed', '-S', type=int, default=42)
    parser.add_argument('--hidden', '-H', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'tanh'])
    parser.add_argument('--model', '-M', type=str, default='MLGCN')
    parser.add_argument('--levels', type=int, default=2)
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')

    parser.add_argument('--wave', type=str, default='haar')
    parser.add_argument('--mode', type=str, default='zero')
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop', choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.3)
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling parameter. Default is 1.0.")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Sparsification parameter. Default is 1e-4.")
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    dataset, data = DataLoader(args.dataset)
    print(f"load {args.dataset} successfully!")
    print('==============================================================')

    warnings.filterwarnings("ignore")


    args_dict = vars(args)
#     args_dict.update(optimized_params)
    args = argparse.Namespace(**args_dict)

    train_rate = 0.6
    val_rate = 0.2
    if args.dataset == 'deezer':
        train_rate = 0.5
        val_rate = 0.25
    # class balance for training dataset
    num_nodes = dataset.num_nodes
    percls_trn = int(round(train_rate * num_nodes / dataset.num_classes))
    val_lb = int(round(val_rate * num_nodes))
    accs, test_accs = [], []

    # 10 times rand part
    for rand in trange(10):
        # training settings
        seed_everything(args.baseseed + rand)
        data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb).to(device)

        model = parse_model(args, dataset, data)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        data, model = data.to(device), model.to(device)
        # model.reset_parameters()
        best_acc = 0.
        final_test_acc = 0.
        es_count = patience = 100
        for epoch in range(500):
            loss, out = train(data)            
            val_acc = valid(data)
            test_acc = test(data)
            if val_acc > best_acc:
                es_count = patience
                best_acc = val_acc
                final_test_acc = test_acc

            else:
                es_count -= 1
            if es_count <= 0:
                break
        accs.append(best_acc)
        test_accs.append(final_test_acc)
    accs = torch.tensor(accs)
    test_accs = torch.tensor(test_accs)

    print(f'{args.dataset} valid_acc: {100 * accs.mean().item():.2f} ± {100 * accs.std().item():.2f}')
    print(f'{args.dataset} test_acc: {100 * test_accs.mean().item():.2f} ± {100 * test_accs.std().item():.2f}')    