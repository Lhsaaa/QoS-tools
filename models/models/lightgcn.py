import torch
from models.abc_model import GeneralGraphRecommender
from models.layers import LightGCNConv
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from torch import nn


class LightGCN(GeneralGraphRecommender):
    r"""LightGCN is a GCN-based recommender model, implemented via PyG.
    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly 
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.label = config["LABEL_FIELD"]

        # load parameters info
        # int type:the embedding size of lightGCN
        self.latent_dim = config['embedding_size']
        # int type:the layer num of lightGCN
        self.n_layers = config['n_layers']
        # float32 type: the weight decay for l2 normalization
        self.reg_weight = config['reg_weight']
        # bool type: whether to require pow when regularization
        self.require_pow = config['require_pow']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.reg_loss = EmbLoss()
        self.loss = nn.L1Loss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(
                all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):

        uid = interaction[self.USER_ID]
        iid = interaction[self.ITEM_ID]
        label = interaction[self.label]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[uid]
        i_embeddings = item_all_embeddings[iid]

        output = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        task_loss = self.loss(output, label)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(uid)
        i_ego_embeddings = self.item_embedding(iid)

        reg_loss = self.reg_loss(
            u_ego_embeddings, i_ego_embeddings, require_pow=self.require_pow)
        loss = task_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return scores
