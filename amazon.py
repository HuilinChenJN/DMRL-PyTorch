'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
import pandas as pd
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy


class AmazonDataset(Dataset):
    def __init__(self, path, split, dataset_name, n_negative=4, check_negative=True, n_folds=0):
        '''
        Amazon Dataset
        :param path: the path of the Dataset
        '''
        super(AmazonDataset, self).__init__()
        self.dataset_name = dataset_name
        self.UserItemNet = None
        self.data_store_path = path
        self.n_users = 0
        self.m_items = 0
        self.a_folds = False
        self.n_folds = n_folds
        if self.n_folds > 0:
            self.a_folds = True
        self.Graph = None
        self.dataMatrix, self.data_num = self.load_rating_file_as_matrix(path + f"/{split}.csv")
        self.textualfeatures, self.imagefeatures, = self.load_features(path)
        # self.num_users, self.num_items = self.trainMatrix.shape
        print('loaded image feature shape', self.imagefeatures.shape )
        print('loaded review feature shape', self.textualfeatures.shape )
        self.get_pairs(self.dataMatrix)
        self.n_negative = n_negative
        self.check_negative = check_negative


    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items, num_total = 0, 0, 0
        df = pd.read_csv(filename, index_col=None, usecols=None)
        for index, row in df.iterrows():
            u, i = int(row['userID']), int(row['itemID'])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        for index, row in df.iterrows():
            user, item, rating = int(row['userID']), int(row['itemID']), 1.0
            if (rating > 0):
                mat[user, item] = 1.0
                num_total += 1
        return mat, num_total

    def load_features(self,data_path):
        import os
        # Prepare textual feture data.
        doc2vec_model = np.load(os.path.join(data_path, 'review.npz'), allow_pickle=True)['arr_0'].item()
        vis_vec = np.load(os.path.join(data_path, 'image_feature.npy'), allow_pickle=True).item()
        filename = data_path + '/train.csv'
        filename_test =  data_path + '/test.csv'
        df = pd.read_csv(filename, index_col=None, usecols=None)
        df_test = pd.read_csv(filename_test, index_col=None, usecols=None)
        asin_i_dic = {}
        trainDataUser, trainDataItem  = [], []
        for index, row in df.iterrows():
            user_id, asin, item_id =int(row['userID']), row['asin'], int(row['itemID'])
            trainDataUser.append(user_id)
            trainDataItem.append(item_id)
            asin_i_dic[item_id] = asin
            self.n_users = max(self.n_users, user_id)
            self.m_items = max(self.m_items, item_id)
        for index, row in df_test.iterrows():
            user_id, asin, item_id =int(row['userID']), row['asin'], int(row['itemID'])
            asin_i_dic[item_id] = asin
            self.n_users = max(self.n_users, user_id)
            self.m_items = max(self.m_items, item_id)
        
        trainDataUser = np.array(trainDataUser)
        trainDataItem = np.array(trainDataItem)

        self.n_users += 1
        self.m_items += 1
        self.UserItemNet = csr_matrix((np.ones(len(trainDataUser)), (trainDataUser, trainDataItem)),
                                    shape=(self.n_users, self.m_items))

        users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        users_D[users_D == 0.] = 1
        items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        items_D[items_D == 0.] = 1.

        features = []
        image_features = []
        if self.dataset_name == 'ToysGames':
            for i in range(self.m_items):
                if asin_i_dic[i] not in doc2vec_model:
                    features.append(np.zeros(512))
                    print(asin_i_dic[i])
                else:
                    features.append(doc2vec_model[asin_i_dic[i]])
                if asin_i_dic[i] not in vis_vec:
                    image_features.append(np.zeros(1024))
                    print(asin_i_dic[i])
                else:
                    image_features.append(np.asarray(vis_vec[asin_i_dic[i]]))
        else:
            for i in range(self.m_items):
                if asin_i_dic[i] not in doc2vec_model:
                    features.append(np.zeros(1024))
                    print(asin_i_dic[i])
                else:
                    features.append(doc2vec_model[asin_i_dic[i]])
                if asin_i_dic[i] not in vis_vec:
                    image_features.append(np.zeros(1024))
                    print(asin_i_dic[i])
                else:
                    # print(len(vis_vec[asin_i_dic[i]]))
                    image_features.append(np.asarray(vis_vec[asin_i_dic[i]]))
        return np.asarray(features,dtype=np.float32),np.asarray(image_features,dtype=np.float32)

    def get_pairs(self, user_item_matrix):
        self.user_item_matrix = lil_matrix(user_item_matrix)
        self.user_item_pairs = np.asarray(self.user_item_matrix.nonzero()).T
        self.user_item_pairs = self.user_item_pairs.tolist()
        self.user_to_positive_set = {u: set(row) for u, row in enumerate(self.user_item_matrix.rows)}

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |0,   R|
            |R^T, 0|
        """
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.data_store_path + '/s_pre_adj_mat.npz'))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"saved norm_mat...")
                sp.save_npz(self.data_store_path + '/s_pre_adj_mat.npz', norm_adj)

            if self.a_folds == True:
                self.Graph = self._split_A_hat(norm_adj, self.n_folds)
                print(f'done split {len(self.Graph)} matrix')
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("don't split the matrix")
        return self.Graph

    def _split_A_hat(self,A, n_folds=50, record_start = 0):
        A_fold = []
        fold_len = (A.shape[0]) // n_folds
        for i_fold in range(n_folds):
            start = i_fold*fold_len
            if i_fold == n_folds - 1:
                end = A.shape[0]
            else:
                end = (i_fold + 1) * fold_len
            self.graph_splict_index.append([start+record_start, end+record_start])
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).cuda()

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, index):
        user_positive_items_pair = deepcopy(self.user_item_pairs[index])
        # sample negative samples
        negative_samples = np.random.randint(
            0,
            self.user_item_matrix.shape[1],
            size=self.n_negative)
        # Check if we sample any positive items as negative samples.
        # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
        # large item set.
        if self.check_negative:
            user = user_positive_items_pair[0]
            for j, neg in enumerate(negative_samples):
                while neg in self.user_to_positive_set[user]:
                    negative_samples[j] = neg = np.random.randint(0, self.user_item_matrix.shape[1])
        # textual and visual features
        textual_feature_pos = self.textualfeatures[user_positive_items_pair[1]]
        visual_feature_pos = self.imagefeatures[user_positive_items_pair[1]]
        textual_feature_neg = self.textualfeatures[negative_samples]
        visual_feature_neg = self.imagefeatures[negative_samples]

        return torch.tensor(user_positive_items_pair), torch.tensor(negative_samples), torch.tensor(textual_feature_pos),\
               torch.tensor(visual_feature_pos), torch.tensor(textual_feature_neg), torch.tensor(visual_feature_neg)


