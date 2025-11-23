from model import *
from modelplus import *

def parse_model(args, dataset, data):
    if args.model == 'MWTGNN':
        model = MWTGNN(dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, args.dropout,
                args.wave)
    elif args.model == 'MLP':
        model = MLP(dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, args.dropout)
    elif args.model == 'GCN':
        model = GCN(dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, args.dropout)
    elif args.model == 'GAT':
        model = GAT(dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, 
                    args.dropout, args.heads)
    elif args.model == 'APPNP':
        model = APPNP_Net(dataset.num_features, dataset.num_classes, args)
    elif args.model == 'JKNet':
        model = GCNJK(dataset.num_features, args.hidden, dataset.num_classes, args.num_layers, args.dropout)
    elif args.model == 'H2GCN':
        model = H2GCN(dataset.num_features, args.hidden, dataset.num_classes, data.edge_index, 
                      dataset.num_nodes, args.num_layers, args.dropout)
    elif args.model == 'GPRGNN':
        model = GPRGNN(dataset.num_features, dataset.num_classes, args)
    elif args.model == 'FAGCN':
        model = FAGCN(dataset.num_features, args.hidden, dataset.num_classes, data, args.dropout, 
                      args.eps, args.num_layers)
    elif args.model == 'GWNN':
        model = GWNN(dataset.num_features, args.hidden, dataset.num_classes, dataset.num_nodes, 
                    wavelets, wavelet_inv, args.dropout)
    
    
    else:
        raise ValueError(f'Invalid method {args.model}')
    return model