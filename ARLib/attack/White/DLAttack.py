import numpy as np
import random
import torch
import os
import scipy.sparse as sp
from copy import deepcopy
from util.tool import targetItemSelect
from util.sampler import next_batch_pairwise
from scipy.sparse import vstack, csr_matrix
from util.loss import l2_reg_loss, bpr_loss
from util.algorithm import find_k_largest

def compute_norm_adj(adj_full: sp.spmatrix) -> sp.csr_matrix:
    """Return normalized adjacency D^{-1/2} A D^{-1/2} as csr_matrix."""
    adj = adj_full.tocoo()
    rowsum = np.array(adj.sum(1)).flatten()
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

class DLAttack():
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
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))
        optimizer = torch.optim.Adam(recommender.model.parameters(), lr=recommender.args.lRate / 10)
        topk = min(recommender.topN)
        p = torch.ones(self.itemNum).cuda() if torch.cuda.is_available() else torch.ones(self.itemNum)
        sigma = 0.8
        for user in self.fakeUser:
            self.fakeUserInject(recommender,user)
            uiAdj = recommender.data.matrix()
            # outer optimization
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

            # prova ad inizializzare il modello con la nuova adjacency
            try:
                tmpRecommender.model._init_uiAdj(adj_full_tmp)
            except Exception:
                # fallback: lascia interaction_mat (model training userà data.norm_adj)
                tmpRecommender.data.interaction_mat = uiAdj2

            tmpRecommender.train(Epoch=self.innerEpoch, optimizer=optimizer, evalNum=5)
            optimizer_attack = torch.optim.Adam(tmpRecommender.model.parameters(), lr=recommender.args.lRate)
            for _ in range(self.outerEpoch):
                with torch.no_grad():
                    Pu, Pi = tmpRecommender.model()
                scores = torch.zeros((uiAdj.shape[0], self.itemNum))
                # if CUDA available, move scores to CUDA to match Pu/Pi device
                if torch.cuda.is_available():
                    scores = scores.cuda()
                for batch in range(0,uiAdj.shape[0], self.batchSize):
                    scores[batch:batch + self.batchSize, :] = (Pu[batch:batch + self.batchSize, :] \
                                    @ Pi.T).detach()
                # mask existing interactions
                nozeroInd = uiAdj2.nonzero()
                # nozeroInd is tuple of arrays (rows, cols)
                scores[nozeroInd[0],nozeroInd[1]] = -10e8
                _, top_items = torch.topk(scores, topk)
                top_items = [[iid.item() for iid in user_top] for user_top in top_items]
                for n, batch in enumerate(next_batch_pairwise(self.data, tmpRecommender.args.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    rec_user_emb, rec_item_emb = tmpRecommender.model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                        neg_idx]
                    users, pos_items, neg_items = [], [], []
                    for idx, u_index in enumerate(list(set(user_idx))):
                        for item in self.targetItem:
                            users.append(u_index)
                            pos_items.append(item)
                            neg_items.append(top_items[u_index][-1])
                    user_emb_cw = Pu[users]
                    pos_items_emb = Pi[pos_items]
                    neg_items_emb = Pi[neg_items]
                    pos_score = torch.mul(user_emb_cw, pos_items_emb).sum(dim=1)
                    neg_score = torch.mul(user_emb_cw, neg_items_emb).sum(dim=1)
                    CWloss = neg_score - pos_score
                    CWloss = CWloss.mean()
                    batch_loss = CWloss + bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(tmpRecommender.args.reg, user_emb,
                                                        pos_item_emb, Pu @ Pi.T)
                    optimizer_attack.zero_grad()
                    batch_loss.backward()
                    optimizer_attack.step()
            with torch.no_grad():
                Pu, Pi = tmpRecommender.model()
            # r is 1D tensor of scores for this user (length = itemNum)
            r = (Pu[user,:] @ Pi.T).detach()
            # apply p weighting (ensure same device)
            if p.device != r.device:
                p = p.to(r.device)
            r = r * p
            # project returns binary vector m and indices index positions
            m, ind = self.project(r, int(self.maliciousFeedbackNum))
            # ensure uiAdj2 is writable: convert to lil then assign, then convert back if needed
            try:
                uiAdj2[user, :] = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
            except Exception:
                # fallback: convert uiAdj2 to dense row assign
                rowarr = uiAdj2.toarray()
                rowarr[user, :] = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                uiAdj2 = sp.csr_matrix(rowarr)
            # update p
            if isinstance(ind, torch.Tensor):
                ind_cpu = ind.cpu().long()
            else:
                ind_cpu = torch.tensor(ind).long()
            # multiply p at those indices
            p[ind_cpu] = p[ind_cpu] * sigma
            if float(max(p)) < 1.0:
                p = torch.ones(self.itemNum).to(p.device)

            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                recommender.data.user_num + self.itemNum, recommender.data.user_num + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:recommender.data.user_num, recommender.data.user_num:] = uiAdj2
            try:
                recommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                recommender.model.data.interaction_mat = uiAdj2
            
            uiAdj = uiAdj2[:, :]
        self.interact = uiAdj
        return self.interact

    def project(self, mat, n):
        """
        Project vector or matrix `mat` onto a binary mask with top-n selections.
        Returns (mask, indices)
        - If mat is 1D tensor: returns mask (1D tensor same shape) and indices (1D LongTensor)
        - If mat is 2D tensor: returns mask (same shape) and indices (LongTensor of shape (n, num_cols))
        Robust to mat being numpy array, scipy sparse row, or torch tensor.
        """
        # Convert input -> torch tensor safely (avoid deepcopy on torch tensors)
        if isinstance(mat, torch.Tensor):
            matrix = mat.detach().clone()
        else:
            # try to convert numpy / sparse / list into tensor
            try:
                # if sparse row/col vector from scipy, convert to dense
                if hasattr(mat, "todense"):
                    arr = np.array(mat.todense()).squeeze()
                else:
                    arr = np.array(mat)
                matrix = torch.tensor(arr, dtype=torch.float32)
            except Exception:
                # ultimate fallback: convert via striding
                matrix = torch.tensor(np.asarray(mat), dtype=torch.float32)

        # 1-D case: choose top-n entries
        if matrix.dim() == 1:
            # if n >= length: return ones
            length = matrix.shape[0]
            if n >= length:
                mask = torch.ones_like(matrix)
                indices = torch.arange(length, dtype=torch.long)
                return mask, indices
            values, indices = torch.topk(matrix, k=n, largest=True, sorted=False)
            mask = torch.zeros_like(matrix)
            mask[indices] = 1.0
            return mask, indices

        # 2-D case: choose top-n along rows for each column (dim=0) - keep original behavior
        elif matrix.dim() == 2:
            rows, cols = matrix.shape
            if n >= rows:
                mask = torch.ones_like(matrix)
                # create indices as all row indices repeated for each column
                indices = torch.stack([torch.arange(rows, dtype=torch.long)[:, None].repeat(1, cols)], dim=1)
                return mask, indices
            # topk along dim=0 yields (k, cols)
            values, indices = torch.topk(matrix, k=n, dim=0, largest=True, sorted=False)
            mask = torch.zeros_like(matrix)
            # indices shape is (n, cols) -> scatter accordingly
            mask.scatter_(0, indices, 1.0)
            return mask, indices

        else:
            # unexpected dim, flatten and operate
            flat = matrix.view(-1)
            if n >= flat.shape[0]:
                mask = torch.ones_like(flat)
                indices = torch.arange(flat.shape[0], dtype=torch.long)
                return mask.view_as(matrix), indices
            values, indices = torch.topk(flat, k=n, largest=True, sorted=False)
            mask = torch.zeros_like(flat)
            mask[indices] = 1.0
            return mask.view_as(matrix), indices

    def fakeUserInject(self, recommender, user):
        Pu, Pi = recommender.model()
        Pu = Pu.detach().cpu()
        Pi = Pi.detach().cpu()


        # aggiungi 1 user (la tua funzione chiamava per singolo user)
        new_user_idx = recommender.data.user_num
        recommender.data.user_num += 1
        recommender.data.user[f"fakeuser{new_user_idx}"] = len(recommender.data.user)
        recommender.data.id2user[len(recommender.data.user) - 1] = f"fakeuser{new_user_idx}"

        # aggiorna training_data: aggiungi targetItem per il fake user corrente
        for i in self.targetItem:
            recommender.data.training_data.append((recommender.data.id2user[user], recommender.data.id2item[i]))

        # ricostruisci interaction_mat user x item
        row, col, entries = [], [], []
        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)

        recommender.data.interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(recommender.data.user_num, recommender.data.item_num),
            dtype=np.float32
        )

        # ---- costruisci adjacency bipartita completa e normalizzala ----
        user_num = recommender.data.user_num
        item_num = recommender.data.item_num
        ui_adj = recommender.data.interaction_mat.tocsr()

        # top/bottom blocks
        adj_top = sp.hstack([sp.csr_matrix((user_num, user_num)), ui_adj], format='csr')
        adj_bottom = sp.hstack([ui_adj.T, sp.csr_matrix((item_num, item_num))], format='csr')
        adj_full = sp.vstack([adj_top, adj_bottom], format='csr')

        # normalizza e salva in data.norm_adj (NCL usa data.norm_adj in train)
        recommender.data.norm_adj = compute_norm_adj(adj_full)

        # reinizializza il recommender (usa i nuovi dati)
        recommender.__init__(recommender.args, recommender.data)

        # ripristina embeddings vecchie se disponibili (copiando le righe esistenti)
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
