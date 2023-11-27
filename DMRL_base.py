import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Module

class DMRL_Base(Module):
    def __init__(self,
                 n_users,
                 n_items,
                 dataset,
                 num_neg=4,
                 n_factors=4,
                 embed_dim=20,
                 decay_r=1e-4,
                 decay_c=1e-3,
                 hidden_layer_dim_a=256,
                 hidden_layer_dim_b=256,
                 dropout_rate_a=0.2,
                 dropout_rate_b=0.2,
                 margin=2,
                 dataset_name='Baby',
                 layer=3,
                 keep_prob = 0.6,
                 is_dropout = True,
                 A_split = False,
                 use_attention = True,
                 use_rank_weight = False
                 ):
        super(DMRL_Base, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_factor = n_factors
        self.embed_dim = embed_dim
        self.num_neg = num_neg
        self.decay_r = decay_r
        self.decay_c = decay_c
        self.hidden_layer_dim_a = hidden_layer_dim_a
        self.hidden_layer_dim_b = hidden_layer_dim_b
        self.dropout_rate_a = dropout_rate_a
        self.dropout_rate_b = dropout_rate_b
        self.dataset_name = dataset_name
        self.keep_prob = keep_prob
        self.is_dropout = is_dropout
        self.n_layers = layer
        self.A_split = A_split
        self.use_attention = use_attention
        self.use_rank_weight = use_rank_weight
        self.Graph = dataset.getSparseGraph()

        # init learnable parameters
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_dim)
        if self.dataset_name == 'ToysGames':
            self.textual_mlp_1 = torch.nn.Sequential(
                torch.nn.Linear(512, 2 * self.hidden_layer_dim_b),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Dropout(self.dropout_rate_b, inplace=False),
            )
        else:
            self.textual_mlp_1 = torch.nn.Sequential(
                torch.nn.Linear(1024, 2 * self.hidden_layer_dim_b),
                torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
                torch.nn.Dropout(self.dropout_rate_b, inplace=False),
            )
        self.visual_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 2 * self.hidden_layer_dim_a),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=False),
            torch.nn.Dropout(self.dropout_rate_a, inplace=False),
        )
        
        self.textual_mlp_2 = torch.nn.Linear(2 * self.hidden_layer_dim_b, self.embed_dim)
        self.visual_mlp_2 = torch.nn.Linear(2 * self.hidden_layer_dim_a, self.embed_dim)

        self.modality_attention_layer = torch.nn.Sequential(
            torch.nn.Linear(int(self.embed_dim * 4 / self.n_factor), 3),
            torch.nn.Tanh(),
            torch.nn.Linear(3, 3, bias=False), 
            nn.Softmax(dim=1)
        )

    def fix_params(self):
        for p in self.textual_mlp_1.parameters():
            p.requires_grad = False
        for p in self.textual_mlp_2.parameters():
            p.requires_grad = False
        for p in self.visual_mlp_1.parameters():
            p.requires_grad = False
        for p in self.visual_mlp_2.parameters():
            p.requires_grad = False
        for p in self.modality_attention_layer.parameters():
            p.requires_grad = False

    def feature_projection_textual(self, textual_feature):
        textual_feature = F.normalize(textual_feature, p=2, dim=1)
        mlp_layer_1 = self.textual_mlp_1(textual_feature)
        output = self.textual_mlp_2(F.normalize(mlp_layer_1, p=2, dim=1))
        return output

    def feature_projection_visual(self, visual_feature):
        visual_feature = F.normalize(visual_feature, p=2, dim=1)
        mlp_layer_1 = self.visual_mlp_1(visual_feature)
        output = self.visual_mlp_2(F.normalize(mlp_layer_1, p=2, dim=1))
        return output

    def _create_distance_correlation(self, x, y):
        def _create_centered_distance(X):
            r = torch.sum(torch.pow(X, 2), 1, keepdim=True)
            D_ = torch.clamp(r - 2 * torch.mm(X, torch.transpose(X, 0, 1)) + torch.transpose(r, 0, 1), min=0.0)
            D = torch.sqrt(D_ + 1e-8)

            D = D - torch.mean(D, 0, keepdim=True) - torch.mean(D, 1, keepdim=True) + torch.mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            n_samples = D1.shape[0]
            dcov_ = torch.clamp(torch.sum(D1 * D2) / (n_samples * n_samples), min=0.0)
            dcov = torch.sqrt(dcov_ + 1e-8)
            return dcov

        D1 = _create_centered_distance(x)
        D2 = _create_centered_distance(y)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        dcor = dcov_12 / (torch.sqrt(torch.clamp(dcov_11 * dcov_22, min=0.0)) + 1e-10)
        return dcor

    def _create_weight(self, user, item, textual, visual):
        input_ = torch.cat((user, item, textual, visual), dim=1)
        input = F.normalize(input_, p=2, dim=1)
        output = self.modality_attention_layer(input)
        return output

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph


    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.is_dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    

    def getEmbedding(self, users, pos_items, neg_items):
        neg_items = neg_items.view(-1)
        # print(users.shape, pos_items.shape, neg_items.shape)
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # obatin the user and item feature
        users_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.item_embedding(pos_items)
        neg_emb_ego = self.item_embedding(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def train_forward(self,
                      user_positive_items_pairs,
                      negative_samples,
                      textual_feature_pos,
                      visual_feature_pos,
                      textual_feature_neg,
                      visual_feature_neg
                      ):



        (users, pos_items, neg_items, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(user_positive_items_pairs[:, 0].long(), 
                                                         user_positive_items_pairs[:, 1].long(), 
                                                         negative_samples.long())

        batch_size = user_positive_items_pairs.shape[0]
        pos_i_t = self.feature_projection_textual(textual_feature_pos)
        pos_i_v = self.feature_projection_visual(visual_feature_pos)

        # negative item embedding (N, K)
        # neg_items = self.item_embedding(negative_samples).view(-1, self.embed_dim)
        neg_i_t = self.feature_projection_textual(textual_feature_neg.view(-1, textual_feature_neg.shape[-1]))
        neg_i_v = self.feature_projection_visual(visual_feature_neg.view(-1, visual_feature_neg.shape[-1]))

        items = torch.cat((pos_items, neg_items), dim=0)
        textual_f = torch.cat((pos_i_t, neg_i_t), dim=0)
        visual_f = torch.cat((pos_i_v, neg_i_v), dim=0)
        user_a_ = users.unsqueeze(1).repeat(1, self.num_neg, 1).view(-1, self.embed_dim)
        user_a = torch.cat((users, user_a_), dim=0)

        factor_emb_dim = int(self.embed_dim / self.n_factor)
        user_factor_embedding = torch.split(users, factor_emb_dim, dim=1)
        item_factor_embedding = torch.split(items, factor_emb_dim, dim=1)
        item_factor_embedding_p = torch.split(pos_items, factor_emb_dim, dim=1)
        textual_factor_embedding = torch.split(textual_f, factor_emb_dim, dim=1)
        textual_factor_embedding_p = torch.split(pos_i_t, factor_emb_dim, dim=1)
        visual_factor_embedding = torch.split(visual_f, factor_emb_dim, dim=1)
        visual_factor_embedding_p = torch.split(pos_i_v, factor_emb_dim, dim=1)

        cor_loss = 0
        for i in range(0, self.n_factor - 1):
            x = visual_factor_embedding_p[i]
            y = visual_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = textual_factor_embedding_p[i]
            y = textual_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = user_factor_embedding[i]
            y = user_factor_embedding[i + 1]
            cor_loss += self._create_distance_correlation(x, y)
            x = item_factor_embedding_p[i]
            y = item_factor_embedding_p[i + 1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factor + 1.0) * self.n_factor / 2)

        p_item, n_item = torch.split(items, [batch_size, self.num_neg * batch_size], dim=0)
        user_ap, user_an = torch.split(user_a, [batch_size, self.num_neg * batch_size], dim=0)

        user_factor_embedding_a = torch.split(user_a, factor_emb_dim, dim=1)
        user_factor_embedding_ap = torch.split(user_ap, factor_emb_dim, dim=1)
        user_factor_embedding_an = torch.split(user_an, factor_emb_dim, dim=1)
        p_item_factor_embedding = torch.split(p_item, factor_emb_dim, dim=1)
        n_item_factor_embedding = torch.split(n_item, factor_emb_dim, dim=1)

        regularizer = 0


        pos_scores, neg_scores = [], []
        for i in range(0, self.n_factor):
            if self.use_attention :
                weights = self._create_weight(user_factor_embedding_a[i], item_factor_embedding[i],
                                            textual_factor_embedding[i], visual_factor_embedding[i])
                p_weights, n_weights = torch.split(weights, [batch_size, self.num_neg * batch_size], dim=0)
            else:
                p_weights = torch.ones(size =(user_factor_embedding_ap[i].shape[0],3)).cuda()
                n_weights = torch.ones(size =(user_factor_embedding_an[i].shape[0],3)).cuda()
            textual_trans = textual_factor_embedding[i]
            p_textual_trans, n_textual_trans = torch.split(textual_trans, [batch_size, self.num_neg * batch_size], dim=0)
            visual_trans = visual_factor_embedding[i]
            p_visual_trans, n_visual_trans = torch.split(visual_trans, [batch_size, self.num_neg * batch_size], dim=0)

            p_score = F.softplus(p_weights[:, 1] * torch.sum(user_factor_embedding_ap[i] * p_textual_trans, 1))\
                      + F.softplus(p_weights[:, 2] * torch.sum(user_factor_embedding_ap[i] * p_visual_trans, 1))\
                      + F.softplus(p_weights[:, 0] * torch.sum(user_factor_embedding_ap[i] * p_item_factor_embedding[i], 1))
            pos_scores.append(p_score.unsqueeze(1))

            n_score =F.softplus(n_weights[:, 1] * torch.sum(user_factor_embedding_an[i] * n_textual_trans, 1))\
                      + F.softplus(n_weights[:, 2] * torch.sum(user_factor_embedding_an[i] * n_visual_trans, 1))\
                      + F.softplus(n_weights[:, 0] * torch.sum(user_factor_embedding_an[i] * n_item_factor_embedding[i], 1))
            neg_scores.append(n_score.unsqueeze(1))

        pos_s = torch.cat(pos_scores, dim=1)
        neg_s = torch.cat(neg_scores, dim=1)

        regularizer += torch.norm(users) ** 2 / 2 + torch.norm(pos_items) ** 2 / 2 + torch.norm(neg_items) ** 2 / 2 \
                      + torch.norm(pos_i_t) ** 2 / 2 + torch.norm(neg_i_t) ** 2 / 2 + torch.norm(pos_i_v) ** 2 / 2 \
                      + torch.norm(neg_i_v) ** 2 / 2 + torch.norm(userEmb0) ** 2 / 2 + torch.norm(posEmb0) ** 2 / 2 \
                      + torch.norm(negEmb0) ** 2 / 2
        regularizer = regularizer / batch_size

        pos_score = torch.sum(pos_s, 1)

        negtive_per_score = torch.sum(neg_s, 1).view(batch_size, self.num_neg)
        negtive_score, _ = torch.max(negtive_per_score, 1)
        loss_per_pair = F.softplus(-(pos_score - negtive_score))

        loss = torch.sum(loss_per_pair)

        return loss + self.decay_c * cor_loss + self.decay_r * regularizer



    def inference_forward(self,
                          user_ids,
                          item_ids,
                          textualfeatures,
                          imagefeatures):
        # (N_USER_IDS, 1, K)
        all_users, all_items = self.computer()


        users = all_users[user_ids].unsqueeze(1)

        # (1, N_ITEM, K)
        items = all_items[item_ids].unsqueeze(0)
        textual = self.feature_projection_textual(textualfeatures).unsqueeze(0)
        visual = self.feature_projection_visual(imagefeatures).unsqueeze(0)

        item_expand = items.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        textual_expand = textual.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        visual_expand = visual.repeat(users.shape[0], 1, 1).view(-1, self.embed_dim)
        users_expand = users.repeat(1, self.n_items, 1).view(-1, self.embed_dim)

        factor_emb_dim = int(self.embed_dim / self.n_factor)
        user_expad_factor_embedding = torch.split(users_expand, factor_emb_dim, dim=1)
        item_expand_factor_embedding = torch.split(item_expand, factor_emb_dim, dim=1)
        textual_expand_factor_embedding = torch.split(textual_expand, factor_emb_dim, dim=1)
        visual_expand_factor_embedding = torch.split(visual_expand, factor_emb_dim, dim=1)

        factor_scores = []
        factor_sc = []
        factor_ws = []
        for i in range(0, self.n_factor):
            weights = self._create_weight(user_expad_factor_embedding[i], item_expand_factor_embedding[i],
                                           textual_expand_factor_embedding[i], visual_expand_factor_embedding[i])
            if self.use_attention :
                weights = self._create_weight(user_expad_factor_embedding[i], item_expand_factor_embedding[i],
                                           textual_expand_factor_embedding[i], visual_expand_factor_embedding[i])
            else:
                weights = torch.ones(size =(user_expad_factor_embedding[i].shape[0],3)).cuda()
            textual_trans = textual_expand_factor_embedding[i]
            visual_trans = visual_expand_factor_embedding[i]
            f_score = F.softplus(weights[:, 1] * torch.sum(user_expad_factor_embedding[i] * textual_trans, 1)) \
                      + F.softplus(weights[:, 2] * torch.sum(user_expad_factor_embedding[i] * visual_trans, 1)) \
                      + F.softplus(weights[:, 0] * torch.sum(user_expad_factor_embedding[i] * item_expand_factor_embedding[i], 1))
            factor_scores.append(f_score.unsqueeze(1))
            factor_sc.append(F.softplus(weights[:, 1] * torch.sum(user_expad_factor_embedding[i] * textual_trans, 1)) \
                      + F.softplus(weights[:, 2] * torch.sum(user_expad_factor_embedding[i] * visual_trans, 1)) \
                      + F.softplus(weights[:, 0] * torch.sum(user_expad_factor_embedding[i] * item_expand_factor_embedding[i], 1)))
            factor_ws.append(weights)

        factor_s = torch.cat(factor_scores, dim=1)
        scores = torch.sum(factor_s, 1).view(users.shape[0], -1)
        return scores, factor_sc, factor_ws
