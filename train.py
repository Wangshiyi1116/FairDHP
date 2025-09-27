from dataset import *
from model import *
from utils import *
from evaluation import *
import argparse
import time
from tqdm import tqdm
import csv
import optuna
from torch import tensor
import warnings
warnings.filterwarnings('ignore')
import math
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from torch.optim.lr_scheduler import ExponentialLR
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, h):
        h = self.lin1(h)
        h = self.lin2(h)
        return h

def run(data, args):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()

    acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if(args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'GCN'):
        # if args.prop == 'scatter':
        #     encoder = GCN_encoder_scatter(args).to(args.device)
        # else:
        encoder = GCN_encoder_spmm(args).to(args.device)
        encoder1 = GCN_encoder_spmm(args).to(args.device)
        encoder2 = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd),
            dict(params=encoder1.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder1.bias, weight_decay=args.e_wd),
            dict(params=encoder2.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder2.bias, weight_decay=args.e_wd)
        ], lr=args.e_lr)
    elif(args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        encoder1 = GCN_encoder_spmm(args).to(args.device)
        encoder2 = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd),
            dict(params=encoder1.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder1.bias, weight_decay=args.e_wd),
            dict(params=encoder2.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder2.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif(args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        encoder1 = GCN_encoder_spmm(args).to(args.device)
        encoder2 = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd),
            dict(params=encoder1.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder1.bias, weight_decay=args.e_wd),
            dict(params=encoder2.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder2.bias, weight_decay=args.e_wd)], lr=args.e_lr)

    vq_adj = VQ_adj(data.x, data.x.shape[1], 100)
    if os.path.isfile(args.dataset+'_mask.pt'):
        print('########## sample already done #############')
        sens_mask = torch.load(args.dataset+'_mask.pt').to(args.device)
        weight = torch.load(args.dataset+'_weight.pt').to(args.device)
    else:
        t_adj = sparse_mx_to_torch_sparse_tensor(data.adj).cuda()
        data.adj = data.adj - sp.eye(data.adj.shape[0])
        # sens_mask = torch.zeros(data.adj.shape)
        sens_mask = sp.coo_matrix(data.adj.shape, dtype=np.float64).tolil()
        h = torch.spmm(t_adj,data.x)
        h1 = torch.zeros(data.x.shape).cuda()
        print('sample begin')
        for i in tqdm(range(data.adj.shape[0])):
            # sens_mask
            neighbor = torch.tensor(data.adj[i].nonzero()).to(args.device)
            mask = (data.sens[neighbor[1].long()] != data.sens[i])
            h_nei_idx = neighbor[1][mask]
            for j in h_nei_idx:
                sens_mask[i, j.item()] = 1
            sens_mask[i, i] = 1
            # weight
            x1 = torch.clone(data.x)
            x1[i][args.sens_idx] = 1-x1[i][args.sens_idx]
            h1[i] = torch.spmm(t_adj,x1)[i]
        weight = log_diff(h1,h)
        print('select done')
        # sens_mask = to_sparse_tensor(sens_mask).cuda()
        sens_mask = sparse_mx_to_torch_sparse_tensor(sens_mask.tocoo()).cuda()
        weight = to_sparse_tensor(weight).cuda()
        torch.save(sens_mask, args.dataset + '_mask.pt')
        torch.save(weight, args.dataset + '_weight.pt')


    data.adj = data.adj + sp.eye(data.adj.shape[0])
    adj_norm = sys_normalized_adjacency(data.adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    data.adj_norm_sp = adj_norm_sp.cuda()

    vq_adj = VQ_adj(data.x,data.x.shape[1],100)
    vq_adj = vq_adj * sens_mask
    sens_adj = to_sparse_tensor(data.adj).cuda() * sens_mask
    vq_adj_norm_sp = sys_normalized_adjacency_torch(vq_adj)
    sens_adj_norm_sp = sys_normalized_adjacency_torch(sens_adj)
    data.vq_adj_norm_sp = vq_adj_norm_sp
    data.sens_adj_norm_sp = sens_adj_norm_sp
    data.weight = weight

    for count in pbar:
        seed_everything(count + args.seed)
        classifier.reset_parameters()
        encoder.reset_parameters()
        encoder1.reset_parameters()
        encoder2.reset_parameters()
        # model.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf

        for epoch in range(0, args.epochs):
            classifier.train()
            encoder.train()
            encoder1.train()
            encoder2.train()
            optimizer_c.zero_grad()
            optimizer_e.zero_grad()

            # h = encoder(data.x + model(data.x), data.edge_index, data.adj_norm_sp)
            h = encoder(data.x, data.edge_index, data.adj_norm_sp)
            h1 = encoder1(data.x, data.edge_index, vq_adj_norm_sp)
            h2 = encoder2(data.x, data.edge_index, sens_adj_norm_sp)
            output = classifier(h+torch.spmm(weight,h1+h2))

            loss_c = F.binary_cross_entropy_with_logits(
                output[data.train_mask], data.y[data.train_mask].unsqueeze(1).to(args.device))

            loss_c.backward()

            optimizer_e.step()
            optimizer_c.step()

            # evaluate classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate(
                data.x, classifier, discriminator, encoder, encoder1, encoder2, data, args)

            if epoch%100 == 0:
                print(epoch, 'Acc:', accs['test'], 'AUC_ROC:', auc_rocs['test'], 'F1:', F1s['test'],
                  'Parity:', tmp_parity['test'], 'Equality:', tmp_equality['test'],'tradeoff:',auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']))


            if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                test_acc = accs['test']
                test_auc_roc = auc_rocs['test']
                test_f1 = F1s['test']
                test_parity, test_equality = tmp_parity['test'], tmp_equality['test']
                best_val_tradeoff = auc_rocs['val'] + F1s['val'] + accs['val'] - (tmp_parity['val'] + tmp_equality['val'])



        acc[count] = test_acc
        f1[count] = test_f1
        auc_roc[count] = test_auc_roc
        parity[count] = test_parity
        equality[count] = test_equality

    res = []
    res.append(args.epochs)
    res.append(args.c_lr)
    res.append(args.c_wd)
    res.append(args.e_lr)
    res.append(args.e_wd)
    res.append(args.hidden)
    res.append('**||**')
    res.append(str(round(np.mean(acc) * 100, 2)) + '±' + str(round(np.std(acc) * 100, 2)))
    res.append(str(round(np.mean(f1) * 100, 2)) + '±' + str(round(np.std(f1) * 100, 2)))
    res.append(str(round(np.mean(auc_roc)* 100,2))+'±'+str(round(np.std(auc_roc) * 100,2)))
    res.append(str(round(np.mean(parity) * 100, 2)) + '±' + str(round(np.std(parity) * 100, 2)))
    res.append(str(round(np.mean(equality) * 100, 2)) + '±' + str(round(np.std(equality) * 100, 2)))


    print('======' + args.dataset + args.encoder + '======')
    print('Acc:'+res[7])
    print('f1:'+res[8])
    print('auc_roc:'+res[9])
    print('parity:'+res[10])
    print('equality:'+res[11])
    with open("24_12_4_{}_{}.csv".format(args.dataset, args.encoder), mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(res)

    return round(np.mean(acc) * 100, 2), round(np.mean(f1) * 100, 2), round(np.mean(auc_roc) * 100, 2), round(np.mean(parity) * 100, 2), round(np.mean(equality) * 100, 2)


def objective(trial):
    args.epochs = trial.suggest_categorical('epochs',[10,50,100,300,500])
    args.c_lr = trial.suggest_categorical('c_lr', [0.1, 0.01, 0.001])
    args.c_wd = trial.suggest_categorical('c_wd', [0, 0.001, 0.0001])
    args.e_lr = trial.suggest_categorical('e_lr', [0.1, 0.01, 0.001])
    args.e_wd = trial.suggest_categorical('e_wd', [0, 0.001, 0.0001])
    args.dropout = trial.suggest_categorical('dropout', [0.2, 0.5, 0.8])
    args.hidden = trial.suggest_categorical('hidden', [12, 16, 32, 64])

    data, args.sens_idx, args.x_min, args.x_max = get_dataset(args.dataset)
    args.num_features, args.num_classes = data.x.shape[1], 2 - 1  # binary classes are 0,1

    args.train_ratio, args.val_ratio = torch.tensor([
        (data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), torch.tensor([
        (data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(
        args.train_ratio) / args.train_ratio, torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[
                                           data.y[data.train_mask].long()], args.val_ratio[data.y[data.val_mask].long()]

    acc, f1, auc_roc, parity, equality = run(data, args)
    return acc, f1, auc_roc, parity, equality




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--d_lr', type=float, default=0.002)
    parser.add_argument('--d_wd', type=float, default=0.0001)
    parser.add_argument('--c_lr', type=float, default=0.1)
    parser.add_argument('--c_wd', type=float, default=0.001)
    parser.add_argument('--e_lr', type=float, default=0.1)
    parser.add_argument('--e_wd', type=float, default=0.001)
    parser.add_argument('--prop', type=str, default='scatter')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=18)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()
    for m_name in ['GCN']:
        args.encoder = m_name
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.init()

        with open("24_12_4_{}_{}.csv".format(args.dataset, args.encoder), mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.ctime(time.time())])
            writer.writerow(['epochs','c_lr','c_wd','e_lr','e_wd','hidden','**||**','acc', 'f1', 'auc_roc', 'DP', 'EO'])

        study = optuna.create_study(directions=['maximize','maximize','maximize',"minimize","minimize"])
        study.optimize(objective, n_trials=300)
        print("总试验次数：", len(study.trials))

        print("帕累托最优解集(Pareto front):")
        trials = sorted(study.best_trials, key=lambda t: t.values)
        for trial in trials:
            print("Trial:", trial.number)
            print(f"Values: FLOPS={trial.values[0]}, accuracy={trial.values[1]}")
            print("Params: ", trial.params, '\n')