import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, JumpingKnowledge, APPNP, MessagePassing
from utils import Dwt, Iwt
from torch.nn import Parameter
from torch_geometric.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul
import scipy.sparse
from torch_geometric.utils import degree
from DWT import DWT_1D, IDWT_1D


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):    
        x = data.x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data):
        x = data.x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return x


class GWNNLayer(nn.Module):
	def __init__(self, node_num, in_channels, out_channels, wavelets, wavelets_inv):
		super(GWNNLayer, self).__init__()
		self.node_num = node_num
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.wavelets = wavelets
		self.wavelets_inv = wavelets_inv

		self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
		self.filter = nn.Parameter(torch.Tensor(self.node_num))

		torch.nn.init.uniform_(self.filter, 0.9, 1.1)
		torch.nn.init.xavier_uniform_(self.weight_matrix)

	def forward(self, features):
		transformed_features = torch.mm(features, self.weight_matrix)
		output = torch.mm(torch.mm(self.wavelets, torch.diag(self.filter)),
						  torch.mm(self.wavelets_inv, transformed_features))
		return output

class GWNN(nn.Module):
    def __init__(self, feature_dims, hidden_dims, output_dims, node_cnt, wavelets, wavelets_inv, dropout_rate):
        super(GWNN, self).__init__()
        self.node_cnt = node_cnt
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.wavelets = wavelets
        self.wavelets_inv = wavelets_inv
        self.dropout_rate = dropout_rate

        self.conv_1 = GWNNLayer(self.node_cnt, self.feature_dims, self.hidden_dims, self.wavelets, self.wavelets_inv)
        self.conv_2 = GWNNLayer(self.node_cnt, self.hidden_dims, self.output_dims, self.wavelets, self.wavelets_inv)
        
    def forward(self, data):
        input = data.x
        input = F.relu(self.conv_1(input))
        output1 = F.dropout(input, training=self.training, p=self.dropout_rate)
        output2 = self.conv_2(output1)
        pred = F.log_softmax(output2, dim=1)
        return pred


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False))

        self.dropout = dropout
        self.activation = F.elu 

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data):
        x = data.x
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return x

class APPNP_Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(num_features, args.hidden)
        self.lin2 = Linear(args.hidden, num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return x

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=not save_mem, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.x
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)

class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = adj_t.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data):
        x = data.x
        # n = data.num_nodes

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha]= 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(num_features, args.hidden)
        self.lin2 = Linear(args.hidden, num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
            return x


class FALayer(MessagePassing):
    def __init__(self, data, num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')
        self.data = data
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.row, self.col = data.edge_index
        self.norm_degree = degree(self.row, num_nodes=data.y.shape[0]).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h):
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
        norm = self.dropout(norm)
        return self.propagate(self.data.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out

class FAGCN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes, data, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(data, num_hidden, dropout))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        h = data.x
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        # return F.log_softmax(h, 1)
        return h

    


class Conv_Pro(MessagePassing):
    '''
    propagation class for mwtgnn
    '''
    def __init__(self, bias=True, **kwargs):
        super(Conv_Pro, self).__init__(aggr='add', **kwargs)


    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j   

    # def __repr__(self):
    #     return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp)


class MWTGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, wave='haar', mode='zero', save_mem=True, use_bn=True):
        super(MWTGNN, self).__init__()

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.prop1 = Conv_Pro()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dwt = DWT_1D(wavename=wave)
        self.iwt = IDWT_1D(wavename=wave)

        self.dropout = dropout
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.wave = wave
        self.mode = mode
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fcs[0](x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        y = self.prop1(x, edge_index)
        
        x_l_in_levels = []
        x_h_in_levels = []

        curr_x_l = x.unsqueeze(1)
        for i in range(1, self.num_layers+1):
            curr_x_l, x_h = self.dwt(curr_x_l)

            x_l_in_levels.append(self.prop1(curr_x_l.squeeze(1), edge_index))
            x_h_in_levels.append(self.prop1(x_h.squeeze(1), edge_index))

        next_x_l = 0
        for i in range(self.num_layers-1, -1, -1):
            curr_x_l = x_l_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()

            curr_x_l = curr_x_l + next_x_l
            next_x_l = self.iwt(curr_x_l.unsqueeze(1), curr_x_h.unsqueeze(1)).squeeze(1)

        x_tag = next_x_l
        assert len(x_l_in_levels) == 0
        
        z0 = y + x_tag
        z = self.bn(self.fcs[-1](z0))

        return z