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
            ui_adj = sp.csr_matrix(([], ([], [])), shape=(
                tmpRecommender.data.user_num + self.itemNum, tmpRecommender.data.user_num + self.itemNum),
                                    dtype=np.float32)
            ui_adj[:tmpRecommender.data.user_num, tmpRecommender.data.user_num:] = uiAdj2
            try:
                tmpRecommender.model._init_uiAdj(ui_adj + ui_adj.T)
            except Exception:
                tmpRecommender.model.data.interaction_mat = uiAdj2
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
        recommender.data.user_num += 1
        recommender.data.user["fakeuser{}".format(recommender.data.user_num)] = len(recommender.data.user)
        recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(recommender.data.user_num)

        row, col, entries = [], [], []
        for i in self.targetItem:
            recommender.data.training_data.append((recommender.data.id2user[user], recommender.data.id2item[i]))
        for pair in recommender.data.training_data:
            row += [recommender.data.user[pair[0]]]
            col += [recommender.data.item[pair[1]]]
            entries += [1.0]

        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                         shape=(recommender.data.user_num, recommender.data.item_num),
                                                         dtype=np.float32)

        recommender.__init__(recommender.args, recommender.data)
        with torch.no_grad():
            try:
                recommender.model.embedding_dict['user_emb'][:Pu.shape[0]] = Pu
                recommender.model.embedding_dict['item_emb'][:] = Pi
            except:
                recommender.model.embedding_dict['user_mf_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], :Pu.shape[1]//2]
                recommender.model.embedding_dict['user_mlp_emb'][:Pu.shape[0]] = Pu[:Pu.shape[0], Pu.shape[1]//2:]
                recommender.model.embedding_dict['item_mf_emb'][:] = Pi[:, :Pi.shape[1]//2]
                recommender.model.embedding_dict['item_mlp_emb'][:] = Pi[:, Pi.shape[1]//2:]

        recommender.model = recommender.model.cuda() if torch.cuda.is_available() else recommender.model

