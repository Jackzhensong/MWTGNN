
        # TEMP = alpha * (1 - alpha) ** np.arange(self.K + 1)
        # TEMP[-1] = (1 - alpha) ** self.K
        # self.temp = Parameter(torch.tensor(TEMP))

        # y = self.prop1(x, edge_index)
        # x_l_in_levels = []
        # x_h_in_levels = []

            # x_l_in_levels.append(self.prop1(curr_x_l.squeeze(1), edge_index))
            # x_h_in_levels.append(self.prop1(x_h.squeeze(1), edge_index))
            # x_l_in_levels.append(curr_x_l.squeeze(1))
            # x_h_in_levels.append(x_h.squeeze(1))



class MLGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dev, num_layers,
                 dropout, wave='haar', mode='zero', save_mem=True, use_bn=True):
        super(MLGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
        # self.bns = nn.ModuleList()
        # self.bns.append(nn.BatchNorm1d(hidden_channels))
        # for _ in range(num_layers - 2):
        #     self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))
        #     self.bns.append(nn.BatchNorm1d(hidden_channels))
        # self.convs.append(GCNConv(hidden_channels, out_channels, cached=not save_mem))

        for i in range(1, num_layers+1):
            self.convs.append(GCNConv(hidden_channels // (2**i), hidden_channels // (2**i), cached=not save_mem))

        # self.lin1 = nn.Linear(in_channels, hidden_channels)
        # self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.bn = nn.BatchNorm1d(out_channels)
  
        self.act = F.relu
        self.dropout = dropout
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.dev = dev
        self.wave = wave
        self.mode = mode
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.bn.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.fcs[0](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.convs[0](x, edge_index))
        
        x_l_in_levels = []
        x_h_in_levels = []
        # shapes_in_levels = []
        dwt = DWT1DForward(J=1, wave=self.wave, mode=self.mode).to(self.dev)
        iwt = DWT1DInverse(wave=self.wave, mode=self.mode).to(self.dev)

        curr_x_l = x.unsqueeze(0)
        for i in range(1, self.num_layers+1):
            curr_x_l, x_h = dwt(curr_x_l)

            x_l_in_levels.append(self.convs[i](curr_x_l, edge_index))
            x_h_in_levels.append(self.convs[i](x_h[0], edge_index))

        next_x_l = 0
        for i in range(self.num_layers-1, -1, -1):
            curr_x_l = x_l_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            # curr_shape = shapes_in_levels.pop()
            curxh = [curr_x_h]
            curr_x_l = curr_x_l + next_x_l
            next_x_l = iwt((curr_x_l, curxh))

        x_tag = next_x_l
        assert len(x_l_in_levels) == 0
        
        x = x + x_tag
        x = self.fcs[-1](x)
        return self.bn(x.squeeze())



        all_z = []
        for i in range(self.num_layers-1, -1, -1):
            curr_x_l = x_l_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()

            output = curr_x_l

            x_high = nn.Softmax(curr_x_h)
            AttMap = torch.mul(output, x_high)
            output = torch.add(output, AttMap)

            # curr_x_l = curr_x_l + next_x_l
            # next_x_l = Iwt(curr_x_l, curr_x_h)
            # next_x_l = self.iwt(curr_x_l.unsqueeze(1), curr_x_h.unsqueeze(1)).squeeze(1)

            all_z = all_z.append(self.lins[i](output))

        x_tag = all_z
        assert len(x_l_in_levels) == 0
        
        x = y + x_tag
        z = self.bn(self.fcs[-1](x))

        return z
    

class MLGCN_trans(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, levels, num_layers,
                 dropout, att_dropout, dev, wave='haar', mode='zero', save_mem=True, use_bn=True):
        super(MLGCN_sim, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        # self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.prop1 = GPR_prop()
        # self.bn = nn.BatchNorm1d(out_channels)

        self.mh_att = MH_ATT(hidden_channels, 1+levels, num_heads, dropout, att_dropout)
        self.pred = MLPs(hidden_channels, hidden_channels, out_channels, num_layers, dropout, 'prelu')
  
        self.act = F.relu
        self.dropout = dropout
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.levels = levels
        self.wave = wave
        self.mode = mode
        self.dev = dev

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.mh_att.reset_parameters()
        self.pred.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_np = x.detach().cpu().numpy()
        dec_x = pywt.swt(x_np, wavelet=self.wave, level=self.levels)

        m_fea = [x]
        for x_l, x_h in reversed(dec_x):
            H_l = self.prop1(torch.from_numpy(x_l).to(self.dev), edge_index).detach().cpu().numpy()
            H_h = self.prop1(torch.from_numpy(x_h).to(self.dev), edge_index).detach().cpu().numpy()
            Y = pywt.iswt([(H_l, H_h)], wavelet=self.wave)
            Y = torch.from_numpy(Y).to(self.dev)
            m_fea.append(Y)
        
        features = torch.cat(m_fea, dim=1)
        features = features.view(data.num_nodes, 1+self.levels, -1)

        att_out = self.mh_att(features)
        out = self.pred(att_out)

        return out


class MLGCNII_sim(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dev, num_layers,
                dropout, wave='haar', mode='zero', save_mem=True, use_bn=True):
        super(MLGCNII_sim, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=not save_mem))

        for i in range(1, num_layers+1):
            self.convs.append(GCNConv(hidden_channels // (2**i), hidden_channels // (2**i), cached=not save_mem))

        # self.lin1 = nn.Linear(in_channels, hidden_channels)
        # self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.bn = nn.BatchNorm1d(out_channels)
  
        self.act = F.relu
        self.dropout = dropout
        self.use_bn = use_bn
        self.num_layers = num_layers
        self.dev = dev
        self.wave = wave
        self.mode = mode
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.bn.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.fcs[0](x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.convs[0](x, edge_index))

        # dwt = DWT1DForward(J=1, wave=self.wave, mode=self.mode).to(self.dev)
        # iwt = DWT1DInverse(wave=self.wave, mode=self.mode).to(self.dev)

        curr_x_l = x
        curr_tmp = [curr_x_l]
        Lists = [curr_tmp]
        for i in range(1, self.num_layers+1):
            list_tmp = []
            for xi in Lists[-1]:
                curr_x_l, x_h = Dwt(xi)
                list_tmp.append(self.convs[i](curr_x_l, edge_index))
                list_tmp.append(self.convs[i](x_h, edge_index))
            Lists.append(list_tmp)

        for i in range(self.num_layers-1, -1, -1):
            list_tmp = []
            x_res = Lists[i]
            for j in range(2**i):
                curr_x_l = Lists[-1].pop()
                curr_x_h = Lists[-1].pop()

                curxh = curr_x_h
                next_x = Iwt(curr_x_l, curxh)
                list_tmp.append(self.convs[i](next_x, edge_index) + x_res[j])
            Lists.append(list_tmp)

        x = Lists[-1][0]

        x = self.fcs[-1](x)
        return self.bn(x)


@torch.no_grad()
def run_full_data(data, forcing=True):
    mask = data.train_mask
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1,keepdim=True)  # Use the class with highest probability.
    if forcing:
        pred = ((data.y.detach() + 1) * mask).view(-1, 1) * mask + (pred + 1) * ~mask
        onehot = torch.zeros((out.shape[0], out.shape[1] + 1), device=Config.device)
        onehot.scatter_(1, pred, 1)
        onehot = onehot[:, 1:]
    else:    #return onehot
        onehot = torch.zeros(out.shape, device=Config.device)
        onehot.scatter_(1, pred, 1)
    return onehot
