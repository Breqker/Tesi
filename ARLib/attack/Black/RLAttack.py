import os
import time
import gym
from gym import spaces
from gym.envs.registration import register
import torch
import torch.nn as nn
from torch import Tensor
from stable_baselines3 import PPO
import numpy as np
import scipy.sparse as sp
import random
from util.tool import targetItemSelect
from util.metrics import AttackMetric
from scipy.sparse import vstack, csr_matrix
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def compute_norm_adj(adj_full):
    """
    adj_full: scipy sparse matrix square (user+item, user+item)
    ritorna la adjacency normalizzata (sp.csr_matrix)
    """
    adj = adj_full.tocoo()
    rowsum = np.array(adj.sum(1)).flatten()
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

class RLAttack():
    def __init__(self, arg, data):
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

        self.env = None
        self.item_num = data.item_num
        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

    def posionDataAttack(self,recommender):
        self.recommender = recommender
        self.fakeUserInject(self.recommender)
        if self.env is None:
            self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
            self.agent = PPO('MlpPolicy', self.env, verbose=1, clip_range=0.1, gamma=1,n_steps=20,n_epochs=10)
            self.agent.learn(total_timesteps=400)
        self.env = MyEnv(self.item_num, self.fakeUser, self.maliciousFeedbackNum, self.recommender, self.targetItem)
        uiAdj = self.recommender.data.matrix()
        obs = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            uiAdj[self.fakeUser[self.env.fakeUserid], :] = 0
            uiAdj[self.fakeUser[self.env.fakeUserid], self.env.itemList] = 1
        self.interact = uiAdj
        return self.interact

    def fakeUserInject(self, recommender):
        Pu, Pi = recommender.model()
        Pu = Pu.detach().cpu()
        Pi = Pi.detach().cpu()

        # aggiorna metadati data con i fake user
        recommender.data.user_num = recommender.data.user_num + self.fakeUserNum
        for i in range(self.fakeUserNum):
            recommender.data.user["fakeuser{}".format(i)] = len(recommender.data.user)
            recommender.data.id2user[len(recommender.data.user) - 1] = "fakeuser{}".format(i)

        self.fakeUser = list(range(self.userNum, self.userNum + self.fakeUserNum))

        # aggiungi interazioni dei fake user al training_data
        for u in self.fakeUser:
            sampleItem = self.targetItem
            for it in sampleItem:
                recommender.data.training_data.append((recommender.data.id2user[u], recommender.data.id2item[it]))

        # ricostruisci interaction_mat user x item con le nuove dimensioni
        row, col, entries = [], [], []
        for pair in recommender.data.training_data:
            row.append(recommender.data.user[pair[0]])
            col.append(recommender.data.item[pair[1]])
            entries.append(1.0)
        recommender.data.interaction_mat = sp.csr_matrix((entries, (row, col)),
                                                        shape=(recommender.data.user_num, recommender.data.item_num),
                                                        dtype=np.float32)

        # reinizializza il recommender con i nuovi data
        recommender.__init__(recommender.args, recommender.data)

        # copia (se possibile) i vecchi pesi nelle nuove embedding
        with torch.no_grad():
            try:
                if Pu is not None and 'user_emb' in recommender.model.embedding_dict:
                    recommender.model.embedding_dict['user_emb'][:Pu.shape[0], :] = Pu
                    recommender.model.embedding_dict['item_emb'][:Pi.shape[0], :] = Pi
            except Exception:
                pass

        # costruisci la matrice bipartita completa (user+item x user+item)
        user_num = recommender.data.user_num
        item_num = recommender.data.item_num
        uiAdj = recommender.data.interaction_mat  # user x item
        ui_adj_full = sp.lil_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
        ui_adj_full[:user_num, user_num:] = uiAdj
        ui_adj_full = ui_adj_full.tocsr()

        # normalizza e aggiorna data.norm_adj
        norm_adj = compute_norm_adj(ui_adj_full)
        recommender.data.norm_adj = norm_adj

        # prova ad aggiornare il modello (LGCN_Encoder._init_uiAdj) o assegnare sparse_norm_adj
        if hasattr(recommender.model, "_init_uiAdj"):
            try:
                recommender.model._init_uiAdj(ui_adj_full)
            except Exception:
                recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(norm_adj).cuda()
        else:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(norm_adj).cuda()

            recommender.model = recommender.model.cuda()




class MyEnv(gym.Env):
    def __init__(self, item_num, fakeUser, maliciousFeedbackNum, recommender, targetItem):
        super(MyEnv, self).__init__()

        self.item_num = item_num
        self.fakeUserNum = len(fakeUser)
        self.recommender = recommender
        self.targetItem = targetItem
        self.maliciousFeedbackNum = maliciousFeedbackNum
        self.fakeUser = fakeUser

        # 定义状态空间和动作空间
        # self.observation_space = spaces.Tuple([spaces.MultiBinary(item_num),spaces.Discrete(self.fakeUserNum)])
        self.observation_space = spaces.MultiBinary(item_num)
        # self.action_space = spaces.Discrete(item_num) # 例如，离散动作空间
        self.action_space = spaces.MultiBinary(item_num) # 例如，离散动作空间

        # self.state_dim = self.observation_space.shape  # feature number of state
        # self.action_dim = self.action_space.n  # feature number of action
    
        self.if_discrete = True
        self.itemList = self.targetItem[:]
        # self.state = (np.array(self.itemList),0)
        self.state = np.zeros(self.item_num)
        self.state[self.itemList] = 1
        self.fakeUserDone = False
        self.fakeUserid = 0

    def reset(self):
        # 重置环境
        self.itemList = self.targetItem[:]
        self.reward = 0
        if self.fakeUserDone:
            self.fakeUserDone = False
            self.fakeUserid = 0
        # self.state = (np.array(self.itemList), self.fakeUserid)
        # self.state = np.array(self.itemList)
        self.state = np.zeros(self.item_num, dtype="int")
        self.state[self.itemList] = 1
        return self.state 

    def step(self, action): 
        ones_indices = np.where(action == 1)[0]
        if len(ones_indices) > self.maliciousFeedbackNum:
            keep_indices = np.random.choice(ones_indices, size=self.maliciousFeedbackNum, replace=False)
            action = np.zeros_like(action, dtype="int")
            action[keep_indices] = 1
        self.state[:] = 0
        self.state[np.where(action == 1)[0]] = 1
        self.state[self.targetItem] = 1
        self.itemList = np.where(self.state == 1)[0]
        self.fakeUserInjectChange(self.recommender, self.fakeUserid, self.itemList)
        optimizer = torch.optim.Adam(self.recommender.model.parameters(), lr= self.recommender.args.lRate / 10)
        self.recommender.train(Epoch=10, optimizer=optimizer, evalNum=1)
        attackmetrics = AttackMetric(self.recommender, self.targetItem, [50])
        reward = attackmetrics.hitRate()[0] * self.recommender.data.user_num
        if self.fakeUserid == self.fakeUserNum - 1: self.fakeUserDone = True
        self.fakeUserid = (self.fakeUserid + 1) % self.fakeUserNum
        info = {}
        return self.state, reward, self.fakeUserDone, info

    def fakeUserInjectChange(self, recommender, fakeUserId, itemList):
        # prendi le dimensioni correnti dai dati
        user_num = recommender.data.user_num
        item_num = recommender.data.item_num

        # copia e aggiorna la matrice user-item
        uiAdj = recommender.data.matrix().copy()
        uiAdj[self.fakeUser[fakeUserId], :] = 0
        uiAdj[self.fakeUser[fakeUserId], itemList] = 1

        # salva su data
        recommender.data.interaction_mat = uiAdj

        # costruisci full bipartite e normalizza
        ui_adj_full = sp.lil_matrix((user_num + item_num, user_num + item_num), dtype=np.float32)
        ui_adj_full[:user_num, user_num:] = uiAdj
        ui_adj_full = ui_adj_full.tocsr()

        norm_adj = compute_norm_adj(ui_adj_full)
        recommender.data.norm_adj = norm_adj

        # aggiorna il modello: preferisci _init_uiAdj, altrimenti assegna sparse_norm_adj
        if hasattr(recommender.model, "_init_uiAdj"):
            try:
                recommender.model._init_uiAdj(ui_adj_full)
            except Exception:
                recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(norm_adj).cuda()
        else:
            recommender.model.sparse_norm_adj = convert_sparse_mat_to_tensor(norm_adj).cuda()


