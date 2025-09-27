import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from copy import deepcopy
from util.tool import targetItemSelect
from util.metrics import AttackMetric


class FedRecAttack():
    def __init__(self, arg, data):
        """
        :param arg: configurazione dei parametri
        :param data: DataLoader
        """
        self.data = data
        self.interact = data.matrix()
        self.userNum = self.interact.shape[0]
        self.itemNum = self.interact.shape[1]

        # Selezione target item
        self.targetItem = targetItemSelect(data, arg)
        self.targetItem = [data.item[i.strip()] for i in self.targetItem]

        self.Epoch = arg.Epoch
        self.innerEpoch = arg.innerEpoch
        self.outerEpoch = arg.outerEpoch

        # Prior knowledge
        self.recommenderGradientRequired = False
        self.recommenderModelRequired = True

        # Limitazioni
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

        self.batchSize = 128

    def posionDataAttack(self, recommender):
        """
        Genera dati avvelenati e aggiorna il modello del recommender
        """
        self.fakeUserInject(recommender)
        uiAdj = recommender.data.matrix()
        topk = min(recommender.topN)
        bestTargetHitRate = -1

        for epoch in range(self.Epoch):
            # Outer optimization
            tmpRecommender = deepcopy(recommender)
            uiAdj2 = uiAdj[:, :]
            ui_adj = sp.csr_matrix(
                ([], ([], [])),
                shape=(self.userNum + self.fakeUserNum + self.itemNum,
                       self.userNum + self.fakeUserNum + self.itemNum),
                dtype=np.float32
            )
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj2
            try:
                tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                tmpRecommender.data.interaction_mat = uiAdj2

            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            
            for _ in range(self.outerEpoch):
                # Ottieni Pu, Pi dal modello temporaneo
                tmpRecommender.train(Epoch=5, optimizer=optimizer_attack, evalNum=5)
                Pu, Pi = tmpRecommender.model()

                # Calcolo dei punteggi
                scores = torch.zeros((self.userNum + self.fakeUserNum, self.itemNum))
                for batch in range(0, self.userNum + self.fakeUserNum, self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] @ Pi.T).detach()
                nozeroInd = uiAdj2.nonzero()
                scores[nozeroInd[0], nozeroInd[1]] = -1e8

                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]

                # Costruzione dati per CW loss
                users, pos_items, neg_items = [], [], []
                for idx, u_index in enumerate(range(self.userNum)):
                    for item in self.targetItem:
                        users.append(u_index)
                        pos_items.append(item)
                        neg_items.append(top_items[u_index].pop())

                user_emb = Pu[users]
                pos_items_emb = Pi[pos_items]
                neg_items_emb = Pi[neg_items]

                pos_score = torch.mul(user_emb, pos_items_emb).sum(dim=1)
                neg_score = torch.mul(user_emb, neg_items_emb).sum(dim=1)
                CWloss = (neg_score - pos_score).mean()

                optimizer_attack.zero_grad()
                CWloss.backward()
                optimizer_attack.step()

            # Aggiornamento interazioni fake user
            for batch in range(0, len(self.fakeUser), self.batchSize):
                uiAdj2[self.fakeUser[batch:batch + self.batchSize], :] = (Pu[self.fakeUser[batch:batch + self.batchSize], :] @ Pi.T).detach().cpu().numpy()

            uiAdj2[self.fakeUser, :] = self.project(uiAdj2[self.fakeUser, :], self.maliciousFeedbackNum)
            for u in self.fakeUser:
                uiAdj2[u, self.targetItem] = 1

            uiAdj = uiAdj2[:, :]

            # Inner optimization
            ui_adj = sp.csr_matrix(
                ([], ([], [])),
                shape=(self.userNum + self.fakeUserNum + self.itemNum,
                       self.userNum + self.fakeUserNum + self.itemNum),
                dtype=np.float32
            )
            ui_adj[:self.userNum + self.fakeUserNum, self.userNum + self.fakeUserNum:] = uiAdj

            try:
                tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                tmpRecommender.data.interaction_mat = uiAdj2
            recommender.train(Epoch=self.innerEpoch, optimizer=optimizer_attack, evalNum=5)

            attackmetrics = AttackMetric(recommender, self.targetItem, [topk])
            targetHitRate = attackmetrics.hitRate()[0]
            print(f"Epoch {epoch+1} Target Hit Rate: {targetHitRate:.4f}")

            if targetHitRate > bestTargetHitRate:
                bestAdj = uiAdj[:, :]
                bestTargetHitRate = targetHitRate

            uiAdj = bestAdj[:, :]
            print(f"BiLevel epoch {epoch + 1} is over\n")

        self.interact = bestAdj
        return self.interact

    def project(self, mat, n):
        """
        Mantiene solo i top-n valori per riga (per i feedback maligni)
        """
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

    def fakeUserInject(self, recommender):
        # Recupera embedding attuali
        Pu, Pi = recommender.model()

        # Aggiorna numero utenti
        recommender.data.user_num += self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user[f"fakeuser{i}"] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = f"fakeuser{i}"

        # Indici utenti falsi
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        # Crea interazioni per utenti falsi
        row, col, entries = [], [], []
        for u in self.fakeUser:
            sampleItem = random.sample(set(range(self.itemNum)), self.maliciousFeedbackNum)
            for i in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[i]))

        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        # Aggiorna matrice di interazione
        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(recommender.data.user_num, recommender.data.item_num),
            dtype=np.float32
        )

        # Ricostruisci il recommender
        recommender.__init__(recommender.args, recommender.data)

        # Aggiorna embedding utente preservando compatibilità dimensionale
        with torch.no_grad():
            if "user_emb" in recommender.model.embedding_dict:
                user_param = recommender.model.embedding_dict["user_emb"]
            elif "user_mf_emb" in recommender.model.embedding_dict:
                user_param = recommender.model.embedding_dict["user_mf_emb"]
            elif "user_mlp_emb" in recommender.model.embedding_dict:
                user_param = recommender.model.embedding_dict["user_mlp_emb"]
            else:
                raise KeyError("Nessuna embedding utente compatibile trovata per l'iniezione.")

            target_dim = user_param.shape[1]

            # Se le dimensioni non coincidono, fai padding con zeri o tronca
            if Pu.shape[1] != target_dim:
                new_Pu = torch.zeros(Pu.shape[0], target_dim, device=Pu.device)
                dim_to_copy = min(Pu.shape[1], target_dim)
                new_Pu[:, :dim_to_copy] = Pu[:, :dim_to_copy]
                Pu = new_Pu

            # Copia embedding utente aggiornate
            user_param[:Pu.shape[0], :] = Pu

        # Manda il modello su GPU se disponibile
        recommender.model = recommender.model.cuda()
