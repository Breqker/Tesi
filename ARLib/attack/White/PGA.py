import numpy as np
import random
import torch
import torch.nn as nn
from util.tool import targetItemSelect
from util.algorithm import find_k_largest
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.loss import bpr_loss, l2_reg_loss
from sklearn.neighbors import LocalOutlierFactor as LOF
from recommender.GMF import GMF
import logging


class PGA():
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

        self.batchSize = 128

    def posionDataAttack(self, recommender):
        Pu, Pi = recommender.model()
        _, maxRecNumItemInd = torch.topk(torch.tensor(self.interact.sum(0)), int(Pi.shape[0] * 0.05))
        self.maxRecNumItemInd = maxRecNumItemInd

        # SGD per warm-up
        optimizer = torch.optim.SGD(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        self.dataUpdate(recommender)
        recommender.__init__(recommender.args, recommender.data)
        newAdj = recommender.data.matrix()

        # Inizializza embedding
        with torch.no_grad():
            for name, emb in recommender.model.embedding_dict.items():
                if "user" in name.lower():
                    dim = min(emb.shape[1], Pu.shape[1])
                    emb[:Pu.shape[0], :dim] = Pu[:, :dim].to(emb.device)
                elif "item" in name.lower():
                    dim = min(emb.shape[1], Pi.shape[1])
                    emb[:Pi.shape[0], :dim] = Pi[:, :dim].to(emb.device)

        self.controlledUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        recommender.train(Epoch=self.Epoch, optimizer=optimizer, evalNum=5)
        originRecommender = deepcopy(recommender)

        # prepara uiAdj con fake user
        uiAdj = newAdj[:, :]
        for u in self.controlledUser:
            uiAdj[u, :] = 0
            uiAdj[u, self.targetItem] = 1
            uiAdj[u, maxRecNumItemInd.cpu().tolist()] = torch.rand([1]).item()

        recommender = deepcopy(originRecommender)
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)

        for epoch in range(self.outerEpoch):
            # outer optimization
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                self.userNum + self.fakeUserNum + self.itemNum, 
                self.userNum + self.fakeUserNum + self.itemNum
            ), dtype=np.float32)
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            try:
                recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                recommender.model.data.interaction_mat = uiAdj

            recommender.train(Epoch=self.Epoch, optimizer=optimizer, evalNum=3)

            # inner optimization (solo se il modello supporta sparse_norm_adj)
            if hasattr(recommender.model, "sparse_norm_adj"):
                tmpRecommender = deepcopy(recommender)

                for _ in range(self.innerEpoch):
                    users, pos_items, neg_items = [], [], []
                    for batch in range(0, self.itemNum, self.batchSize):
                        ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                            self.userNum + self.fakeUserNum + self.itemNum,
                            self.userNum + self.fakeUserNum + self.itemNum
                        ), dtype=np.float32)
                        ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj
                        tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)

                        try:
                            tmpRecommender.model.sparse_norm_adj.requires_grad = True
                        except Exception:
                            pass

                        Pu, Pi = tmpRecommender.model()

                        if len(users) == 0:
                            scores = torch.matmul(Pu, Pi.transpose(0, 1))
                            _, top_items = torch.topk(scores, 50)
                            top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                            for idx, u_index in enumerate(list(range(self.userNum))):
                                for item in self.targetItem:
                                    users.append(u_index)
                                    pos_items.append(item)
                                    neg_items.append(top_items[u_index].pop())

                        # compute CW loss
                        user_emb = Pu[users]
                        pos_items_emb = Pi[pos_items]
                        neg_items_emb = Pi[neg_items]
                        pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                        neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                        CWloss = (neg_score - pos_score).mean()
                        Loss = CWloss

                        # gradiente
                        doubleGrad = torch.autograd.grad(Loss, tmpRecommender.model.sparse_norm_adj, allow_unused=True)[0]
                        if doubleGrad is None:
                            doubleGrad = torch.zeros_like(tmpRecommender.model.sparse_norm_adj.to_dense())

                        # aggiorna uiAdj
                        with torch.no_grad():
                            rowsum = np.array((ui_adj + ui_adj.T).sum(1))
                            d_inv = np.power(rowsum, -0.5).flatten()
                            d_inv[np.isinf(d_inv)] = 0.
                            d_mat_inv = sp.diags(d_inv)
                            indices = torch.tensor([list(range(d_mat_inv.shape[0])), list(range(d_mat_inv.shape[0]))])
                            values = torch.tensor(d_inv, dtype=torch.float32)
                            if torch.cuda.is_available():
                                indices = indices.cuda()
                                values = values.cuda()
                            d_mat_inv = torch.sparse_coo_tensor(indices=indices, values=values,
                                                                size=[d_mat_inv.shape[0], d_mat_inv.shape[0]]).to(doubleGrad.device)
                            norm_adj_tmp = torch.sparse.mm(d_mat_inv, doubleGrad)
                            doubleGrad = torch.sparse.mm(norm_adj_tmp, d_mat_inv)
                            doubleGrad = doubleGrad.to_dense()
                            grad = doubleGrad[
                                :self.userNum + self.fakeUserNum,
                                self.userNum + self.fakeUserNum:][self.controlledUser, :] + doubleGrad[
                                                                                            self.userNum + self.fakeUserNum:,
                                                                                            :self.userNum + self.fakeUserNum].T[
                                                                                            self.controlledUser, :]
                            subMatrix = torch.tensor(uiAdj[self.controlledUser, :].todense()).cuda()
                            subMatrix -= 0.2 * torch.tanh(grad)
                            subMatrix[subMatrix > 1] = 1
                            subMatrix[subMatrix <= 0] = 1e-8
                            uiAdj[self.controlledUser, :] = subMatrix.cpu()

                        print(">> batchNum:{} Loss:{}".format(int(batch / self.batchSize), Loss))

            # proietta i top-n feedback
            uiAdj[self.controlledUser, :] = self.project(uiAdj[self.controlledUser, :],
                                                        int(self.maliciousFeedbackSize * self.itemNum))
            for u in self.controlledUser:
                for i in self.targetItem:
                    uiAdj[u, i] = 1

            print("attack step {} is over\n".format(epoch + 1))

        self.interact = uiAdj
        return self.interact

    def project(self, mat, n):
        try:
            matrix = torch.tensor(mat[:, :].todense())
            _, indices = torch.topk(matrix, n, dim=1)
            matrix.zero_()
            matrix.scatter_(1, indices, 1)
        except:
            matrix = mat[:, :]
            for i in range(matrix.shape[0]):
                subMatrix = torch.tensor(matrix[i, :].todense())
                topk_values, topk_indices = torch.topk(subMatrix, n)
                subMatrix.zero_()
                subMatrix[0, topk_indices] = 1
                matrix[i, :] = subMatrix[:, :].flatten()
        return matrix

    def dataUpdate(self, recommender):
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)
        n_nodes = recommender.data.user_num + recommender.data.item_num
        row_idx = [recommender.data.user[pair[0]] for pair in recommender.data.training_data]
        col_idx = [recommender.data.item[pair[1]] for pair in recommender.data.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + recommender.data.user_num)), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T
        recommender.data.ui_adj = adj_mat
        recommender.data.norm_adj = recommender.data.normalize_graph_mat(recommender.data.ui_adj)
        row, col, entries = [], [], []
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]
        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X, device=None, requires_grad=True):
        """
        Convert a scipy sparse matrix X (COO) to a torch sparse_coo_tensor.
        - device: torch.device or string ('cpu'/'cuda') or None (defaults to current device).
        - requires_grad: if True, the values tensor will have requires_grad=True so autograd can flow.
        """
        coo = X.tocoo()
        # indices: 2 x nnz, type long
        # ensure int64 for indices
        row_idx = coo.row.astype(np.int64)
        col_idx = coo.col.astype(np.int64)
        indices = torch.LongTensor([row_idx, col_idx])
        # values: float tensor; create as leaf and set requires_grad if requested
        values = torch.from_numpy(coo.data.astype(np.float32))
        if requires_grad:
            # make a leaf tensor with requires_grad True so autograd can track it
            values = values.clone().detach().requires_grad_(True)
        # send to device if specified
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        indices = indices.to(device)
        values = values.to(device)
        # create sparse_coo_tensor (preferred over deprecated constructors)
        sparse = torch.sparse_coo_tensor(indices, values, coo.shape, device=device)
        return sparse
