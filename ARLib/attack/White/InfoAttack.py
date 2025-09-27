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
import logging

def compute_norm_adj(adj_full: sp.spmatrix) -> sp.csr_matrix:
    """Return normalized adjacency D^{-1/2} A D^{-1/2} as csr_matrix."""
    rowsum = np.array(adj_full.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt.dot(adj_full).dot(D_inv_sqrt)
    return norm_adj.tocsr()

def convert_sparse_mat_to_tensor(X: sp.spmatrix):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

class InfoAttack():
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

        self.batchSize = 256


    def posionDataAttack(self, recommender):
        Pu, Pi = recommender.model()
        with torch.no_grad():
            view1 = Pi[:, :].detach()
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        bestTargetHitRate = -1
        ind = None
        for epoch in range(self.Epoch):
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            # costruisci full adjacency per tmpRecommender e aggiorna norm_adj
            user_num_tmp = tmpRecommender.data.user_num
            item_num_tmp = tmpRecommender.data.item_num
            adj_top_tmp = sp.hstack([sp.csr_matrix((user_num_tmp, user_num_tmp)), uiAdj2], format='csr')
            adj_bottom_tmp = sp.hstack([uiAdj2.T, sp.csr_matrix((item_num_tmp, item_num_tmp))], format='csr')
            adj_full_tmp = sp.vstack([adj_top_tmp, adj_bottom_tmp], format='csr')

            # aggiorna data.norm_adj usata dal modello (NCL.train legge self.data.norm_adj)
            tmpRecommender.data.norm_adj = compute_norm_adj(adj_full_tmp)

            # prova ad inizializzare il modello con la nuova adjacency (fallback se il metodo non esiste)
            try:
                tmpRecommender.model._init_uiAdj(adj_full_tmp)
            except Exception:
                tmpRecommender.data.interaction_mat = uiAdj2

            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                Pu, Pi = tmpRecommender.model()
                scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
                for batch in range(0,self.userNum + self.fakeUserNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                nozeroInd = uiAdj2.nonzero()   # tuple (rows, cols)
                scores[nozeroInd[0], nozeroInd[1]] = -1e9
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
                CWLoss = neg_score - pos_score
                CWLoss = CWLoss.mean()
                
                InfoLoss = 0
                k = 0
                for batch in range(0,self.itemNum, self.batchSize):
                    k += 1
                    view2 = Pi[batch:batch + self.batchSize, :]
                    InfoLoss += self.InfoNCEBatch(view1, view2, 0.2, batch, self.batchSize).mean()
                InfoLoss = InfoLoss/k
                with torch.no_grad():
                    self.sum = InfoLoss + CWLoss
                    self.a = CWLoss/self.sum
                    self.b = InfoLoss/self.sum
                Loss = self.a * CWLoss + self.b * InfoLoss
                print("loss:{}".format(Loss))
                optimizer_attack.zero_grad()
                Loss.backward()
                optimizer_attack.step()

            Pu, Pi = tmpRecommender.model()
            for batch in range(0,len(self.fakeUser),self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()
            uiAdj2[self.fakeUser, :],_ = self.relaxProject(uiAdj2[self.fakeUser, :],
                                                          self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u,self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, self.userNum + self.fakeUserNum + self.itemNum),
                                   dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            try:
                recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                recommender.model.data.interaction_mat = uiAdj2
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=1)

            attackmetrics = AttackMetric(recommender, self.targetItem, [topk])
            targetHitRate = attackmetrics.hitRate()[0]
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
        return matrix, indices

    def relaxProject(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, 2*n, dim=1)
            newIndices = torch.zeros((matrix.shape[0],n))
            for i in range(newIndices.shape[0]):
                newIndices[i,:] = indices[i,random.sample(list(range(2*n)),n)]
            indices = newIndices[:,:]
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:,:]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, 2*n)
                subMatrix.zero_()  
                subMatrix[0, topk_indices[0,random.sample(list(range(2*n)),n)]] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix, indices

    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        Pu = Pu.detach().cpu()
        Pi = Pi.detach().cpu()


        # aggiorna user_num e mappe
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user[f"fakeuser{i}"] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = f"fakeuser{i}"

        # aggiorna lista fake users
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        # costruisci training_data -> interaction_mat
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem = random.sample(range(self.itemNum), self.maliciousFeedbackNum)
            for it in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[it]))
        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(recommender.data.user_num, recommender.data.item_num),
            dtype=np.float32
        )

        # costruisci adjacency bipartita completa e normalize (user+item)
        user_num = recommender.data.user_num
        item_num = recommender.data.item_num
        ui_adj = recommender.data.interaction_mat.tocsr()
        adj_top = sp.hstack([sp.csr_matrix((user_num, user_num)), ui_adj], format='csr')
        adj_bottom = sp.hstack([ui_adj.T, sp.csr_matrix((item_num, item_num))], format='csr')
        adj_full = sp.vstack([adj_top, adj_bottom], format='csr')
        recommender.data.norm_adj = compute_norm_adj(adj_full)

        # reinizializza il recommender (user_num aggiornato)
        recommender.__init__(recommender.args, recommender.data)

        # ripristina embeddings vecchie se possibile (copia righe presenti)
        with torch.no_grad():
            if Pu is not None and 'user_emb' in recommender.model.embedding_dict:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0], :] = Pu
            if Pi is not None and 'item_emb' in recommender.model.embedding_dict:
                recommender.model.embedding_dict['item_emb'][:Pi.shape[0], :] = Pi

        # aggiorna sparse_norm_adj
        try:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(recommender.data.norm_adj).cuda()
        except Exception:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(recommender.data.norm_adj)
        
        recommender.model = recommender.model.cuda()


    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return cl_loss

    def InfoNCEBatch(self, view1, view2, temperature, batch, batchSize):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1[batch:batch+batchSize,:] * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=0)
        cl_loss = -torch.log(pos_score / ttl_score)
        return cl_loss
