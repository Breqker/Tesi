import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from util.tool import targetItemSelect
from util.metrics import AttackMetric
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy

def compute_norm_adj(interact_mat):
    """Costruisce e normalizza la matrice di adiacenza."""
    num_users, num_items = interact_mat.shape
    A = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)

    A[:num_users, num_users:] = interact_mat
    A[num_users:, :num_users] = interact_mat.T

    degree = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    return (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def convert_sparse_mat_to_tensor(sparse_mx):
    """Converte scipy.sparse in torch.sparse_coo_tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)

class PipAttack():
    def __init__(self, arg, data):
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]
        self.realuserNum = self.userNum
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # Limitation
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.itemNum)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.batchSize = 2048
        # FIX: la MLP deve essere dimensionata solo sui real user
        self.popularity_model = MLP(self.realuserNum)
        self.alpha = 0.1
        self.init_popularity_mlp()

    def init_popularity_mlp(self):
        n = int(self.itemNum * 0.2)
        sorteditem = np.argsort(self.interact.sum(axis=0))
        popular_items = sorteditem[-n:]

        labels = np.zeros(self.itemNum)
        labels[popular_items] = 1

        # FIX: input della MLP sono solo real user
        popular_data = torch.tensor(self.interact.T.todense(), dtype=torch.float32)
        one_hot_labels = np.zeros((self.itemNum, 2))
        for idx in range(self.itemNum):
            one_hot_labels[idx] = [1,0] if labels[idx]==0 else [0,1]
        labels = torch.tensor(one_hot_labels, dtype=torch.float32)

        dataset = TensorDataset(popular_data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.popularity_model.parameters(), lr=0.001)

        for epoch in range(10):
            for inputs, lbl in train_loader:
                optimizer.zero_grad()
                outputs = self.popularity_model(inputs)
                loss = criterion(outputs, lbl)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1} popularity loss: {loss.item():.3f}")

    def posionDataAttack(self, recommender):
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate/10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        ind = None
        for epoch in range(self.Epoch):
            # outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj2
            try:
                tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                tmpRecommender.model.data.interaction_mat = uiAdj2
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                Pu, Pi = tmpRecommender.model()
                scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
                for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                nozeroInd = uiAdj2.nonzero()
                scores[nozeroInd[0],nozeroInd[1]] = -10e8
                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                users, pos_items, neg_items = [], [], []
                for idx, u_index in enumerate(list(range(self.userNum))):
                    for item in self.targetItem:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())
                user_emb = Pu[users]
                pos_items_emb = Pi[pos_items]
                neg_items_emb = Pi[neg_items]
                pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                ExplicitPromotionLoss = - pos_score
                ExplicitPromotionLoss = ExplicitPromotionLoss.mean()
                
                criterion = nn.CrossEntropyLoss()
                popular_data = torch.tensor(self.interact.T.todense(), dtype=torch.float32)
                outputs = self.popularity_model(popular_data[self.targetItem, :self.realuserNum])

                one_hot_labels = np.zeros((len(self.targetItem), 2))
                for idx in range(len(self.targetItem)):
                    one_hot_labels[idx] = [0, 1]
                labels = torch.tensor(one_hot_labels, dtype=torch.float32)
                PopularityPromotionLoss = criterion(outputs, labels)
                

                lossall = ExplicitPromotionLoss + self.alpha * PopularityPromotionLoss
                optimizer_attack.zero_grad()
                lossall.backward()
                optimizer_attack.step()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :], _ = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]
            # for batch in range(0,len(self.fakeUser),self.batchSize):
            #     uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] \
            #                         @ Pi.T).detach().cpu().numpy()
            
            # if ind is None:
            #     uiAdj2[self.fakeUser, :], ind = self.project(uiAdj2[self.fakeUser, :],
            #                                     ([self.maliciousFeedbackNum//self.Epoch] * (self.Epoch - self.maliciousFeedbackNum%self.Epoch) \
            #                                     + [self.maliciousFeedbackNum//self.Epoch + 1] * (self.maliciousFeedbackNum%self.Epoch))[epoch])
            # else:
            #     for step, u in enumerate(self.fakeUser):
            #         uiAdj2[u,ind[step]] = -10e9
            #     uiAdj2[self.fakeUser, :], indCurrent = self.project(uiAdj2[self.fakeUser, :],
            #                                                 ([self.maliciousFeedbackNum//self.Epoch] * (self.Epoch - self.maliciousFeedbackNum%self.Epoch) \
            #                                     + [self.maliciousFeedbackNum//self.Epoch + 1] * (self.maliciousFeedbackNum%self.Epoch))[epoch])
            #     for step, u in enumerate(self.fakeUser):
            #         uiAdj2[u,ind[step]] = 1
            #     ind = torch.cat((ind,indCurrent), dim=1)
            # for u in self.fakeUser:
            #     uiAdj2[u,self.targetItem] = 1
            # uiAdj = uiAdj2[:, :]

            # inner optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            try:
                recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                recommender.model.data.interaction_mat = uiAdj2
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)

            attackmetrics = AttackMetric(recommender, self.targetItem, [topk])
            targetHitRate = attackmetrics.hitRate()[0]
            print(targetHitRate)
            if targetHitRate > bestTargetHitRate:
                bestAdj = uiAdj[:,:]
                bestTargetHitRate = targetHitRate

            uiAdj = bestAdj[:,:]

            print("BiLevel epoch {} is over\n".format(epoch + 1))
        self.interact = bestAdj
        return self.interact

    def project(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, n, dim=1)
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:,:]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, n)
                subMatrix.zero_()  
                subMatrix[0, topk_indices] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix,indices

    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        Pu = Pu.detach().cpu()
        Pi = Pi.detach().cpu()

        # aggiorna numero utenti
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user[f"fakeuser{i}"] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = f"fakeuser{i}"

        # lista di fake user
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        # aggiorna interazioni
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem = random.sample(range(self.itemNum), self.maliciousFeedbackNum)
            for it in sampleItem:
                recommender.data.training_data.append(
                    (recommender.data.id2user[u], recommender.data.id2item[it])
                )
        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(recommender.data.user_num, recommender.data.item_num),
            dtype=np.float32
        )

        # costruisci adjacency completa e normalizzata
        adj_full = compute_norm_adj(recommender.data.interaction_mat)
        recommender.data.norm_adj = adj_full

        # reinizializza il recommender
        recommender.__init__(recommender.args, recommender.data)

        # rialloca embeddings
        with torch.no_grad():
            try:
                if Pu is not None and 'user_emb' in recommender.model.embedding_dict:
                    recommender.model.embedding_dict['user_emb'][:Pu.shape[0], :] = Pu
                    recommender.model.embedding_dict['item_emb'][:Pi.shape[0], :] = Pi
                else:
                    # fallback per NCF/MLP
                    for key in ['user_mf_emb','user_mlp_emb','item_mf_emb','item_mlp_emb']:
                        if key in recommender.model.embedding_dict:
                            old = recommender.model.embedding_dict[key]
                            new = torch.zeros(
                                (recommender.data.user_num if 'user' in key else recommender.data.item_num,
                                old.shape[1])
                            ).to(old.device)
                            new[:old.shape[0]] = old
                            recommender.model.embedding_dict[key] = nn.Parameter(new)
            except Exception:
                pass

        # aggiorna sparse_norm_adj
        try:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(adj_full).cuda()
        except Exception:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(adj_full)

        recommender.model = recommender.model.cuda()
