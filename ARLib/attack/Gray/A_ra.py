import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from util.algorithm import find_k_largest
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from recommender.GMF import GMF
import logging

def compute_norm_adj(adj_full):
    rowsum = np.array(adj_full.sum(1)).flatten()
    rowsum_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    rowsum_inv_sqrt[np.isinf(rowsum_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(rowsum_inv_sqrt)
    norm_adj = D_inv_sqrt.dot(adj_full).dot(D_inv_sqrt)
    return norm_adj.tocsr()

def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

class A_ra():
    def __init__(self, arg, data):
        """
        :param arg: parameter configuration
        :param data: dataLoder
        """
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]
        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # capability prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # limitation 
        self.maliciousUserSize = arg.maliciousUserSize
        self.maliciousFeedbackSize = arg.maliciousFeedbackSize
        if self.maliciousFeedbackSize == 0:
            self.maliciousFeedbackNum = int(self.interact.sum() / data.user_num)
        elif self.maliciousFeedbackSize >= 1:
            self.maliciousFeedbackNum = self.maliciousFeedbackSize
        else:
            self.maliciousFeedbackNum = int(self.maliciousFeedbackSize * self.item_num)

        if self.maliciousUserSize < 1:
            self.fakeUserNum = int(data.user_num * self.maliciousUserSize)
        else:
            self.fakeUserNum = int(self.maliciousUserSize)

        self.sigma = 1
        self.n = 100
        self.batchSize = 128

    def posionDataAttack(self, recommender):
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
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
                tmpRecommender.data.interaction_mat = uiAdj2
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                # Pu, Pi = tmpRecommender.model()
                # We do not know Pu, so learn Pu
                # Seleziona embedding utente corretta
                if "user_emb" in tmpRecommender.model.embedding_dict:
                    emb = tmpRecommender.model.embedding_dict["user_emb"]
                elif "user_mf_emb" in tmpRecommender.model.embedding_dict:
                    emb = tmpRecommender.model.embedding_dict["user_mf_emb"]
                elif "user_mlp_emb" in tmpRecommender.model.embedding_dict:
                    emb = tmpRecommender.model.embedding_dict["user_mlp_emb"]
                else:
                    raise KeyError("Nessuna embedding utente trovata nel modello!")

                optimizer = torch.optim.Adam([emb], lr=recommender.args.lRate)
                tmpRecommender.train(Epoch=5, optimizer=optimizer, evalNum=5)

                _, Pi = tmpRecommender.model()
                approximateUserEmb = (torch.randn((self.n, Pi.shape[1])) * self.sigma).cuda()
                loss = 0
                for i in self.targetItem:
                    for j in range(self.n):
                        loss += -torch.log(torch.sigmoid(approximateUserEmb[j, :] @ Pi[i, :].T) + 10e-8)
                loss = loss.mean()

                optimizer_attack.zero_grad()
                loss.backward()
                optimizer_attack.step()
            Pu, Pi = tmpRecommender.model()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :] = self.project(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

            # inner optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj
            try:
                recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                recommender.data.interaction_mat = uiAdj

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
        return matrix

    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        recommender.data.user_num += self.fakeUserNum

        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        # Aggiorna interazioni dei fake user
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem = random.sample(set(range(self.itemNum)), self.maliciousFeedbackNum)
            for i in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[i]))
        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)), shape=(recommender.data.user_num, recommender.data.item_num), dtype=np.float32
        )

        # Ricostruisci matrice bipartita completa (user+item)
        uiAdj = recommender.data.interaction_mat
        user_num = recommender.data.user_num
        item_num = recommender.data.item_num
        ui_adj_full = sp.lil_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
        ui_adj_full[:user_num, user_num:] = uiAdj
        ui_adj_full = ui_adj_full.tocsr()

        # normalizza e aggiorna data.norm_adj
        recommender.data.norm_adj = compute_norm_adj(ui_adj_full)

        # reinizializza modello
        recommender.__init__(recommender.args, recommender.data)

        with torch.no_grad():
            # copia embeddings vecchie
            if 'user_emb' in recommender.model.embedding_dict:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
            if 'item_emb' in recommender.model.embedding_dict:
                recommender.model.embedding_dict['item_emb'][:Pi.shape[0]] = Pi

        # aggiorna sparse_norm_adj

        recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(recommender.data.norm_adj).cuda()
   
        recommender.model = recommender.model.cuda()
