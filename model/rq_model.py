
import torch
from torch import nn, optim
from torch.nn import functional as F

from model.Huffman import huffmanTree
from model.Huffman import Draw_RBTree


class ModelRQuery(nn.Module):
    """
    token-level attention for passage-level relation extraction.
    """

    def __init__(self,
                 sentences_encoder,
                 num_class,
                 rel2id):
        """
        Args:
            sentences_encoder: encoder for whole passage (bag of sentences)
            num_class: number of classes
        """
        super().__init__()
        self.sentences_encoder = sentences_encoder
        self.embed_dim = self.sentences_encoder.hidden_size
        self.num_class = num_class
        self.fc1 = nn.Linear(self.embed_dim*2, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.relation_embeddings = nn.Parameter(torch.empty(self.num_class, self.embed_dim))
        nn.init.xavier_normal_(self.relation_embeddings)
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.act=nn.Tanh()
        # self.dropout=nn.Dropout(0.1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel


    def FC1(self,x,isAct):

        # x=self.dropout(x)
        if isAct:
            x=self.act(x)
        x=self.fc1(x)
        return x

    def FC2(self,x,isAct):

        # x=self.dropout(x)
        if isAct:
            x=self.act(x)
        x=self.fc2(x)
        return x

    def Construct_Huffman_Trees(self, node_feature,node_distance):
        node_weight = self.softmax(node_distance)
        input = []
        for i in range(node_feature.shape[0]):
                node=[]
                node.append(str(i))
                node.append(node_distance[i])
                node.append(node_feature[i,:])
                node.append(node_weight[i])
                node=tuple(node)
                input.append(node)
        output=huffmanTree(patt=self,num=input)
        tree=output.getTree()
        return  tree.feature   #relation_scores, relation_features


    def Function(self,bag):

        batch_features=[]

        for i in range(len(bag)):
            node_distance=F.cosine_similarity(bag[i].unsqueeze(1), bag[i].unsqueeze(0),dim=-1,eps=1e-08)
            node_distance=node_distance.sum(axis=0)
            feature = self.Construct_Huffman_Trees(bag[i], node_distance)
            batch_features.append(feature)

        batch_features = torch.stack(batch_features, dim=0)
        # print(1)
        return batch_features


    def forward(self, token, mask,nums, train=True):
        """
        Args:
            token: (nsum, L), index of tokens
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        start=0
        end=0
        bag=[]
        batch_size = len(nums)
        # max_len = token.shape[-1]
        if mask is not None:
            rep = self.sentences_encoder(token, mask)  # (B, max_len, H)
            for i in range(batch_size):
                end=end+nums[i]
                bag.append(rep[start:end])
                start=start+nums[i]

            # rep = rep.reshape(batch_size,token.shape[1],rep.shape[1],rep.shape[2])
        else:
            rep = self.sentences_encoder(token)  # (nsum, H)
            for i in range(batch_size):
                end = end + nums[i]
                bag.append(rep[start:end])
                start = start + nums[i]
        if train:

            batch_logits =self.Function(bag)
            batch_scores=torch.matmul(batch_logits,self.relation_embeddings.permute(1,0))
            rel_scores = self.sigm(batch_scores)  # (B, N, H) -> (B, N, 1) -> (B, N)


        else:
            with torch.no_grad():

                batch_logits =self.Function(bag)
                batch_scores=torch.matmul(batch_logits,self.relation_embeddings.permute(1,0))
                rel_scores = self.sigm(batch_scores)  # (B, N, H) -> (B, N, 1) -> (B, N)

        return rel_scores
