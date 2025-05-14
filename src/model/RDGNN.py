import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(dim))
        self.b_2 = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class RDGNN(nn.Module):
    def __init__(self, args, K):
        super().__init__()
        self.max_tree_dis = args.max_tree_dis
        self.gnn_input_dim = args.bart_dim
        self.gnn_hidden_dim = args.gnn_output_dim
        self.bart_drop = nn.Dropout(args.bart_dropout)
        self.gnn_drop = nn.Dropout(args.gnn_dropout)
        self.layernorm = LayerNorm(args.bart_dim)
        self.abl_mode = args.abl_mode
        if self.abl_mode != 'type':
            self.dep_embedding = nn.Embedding(args.dep_vocab_size, args.dep_embed_dim)
            self.dep_imp_function = DEP_IMP(args.dep_embed_dim)
        self.gnn_layer_num = args.gnn_layer_num

        self.W = nn.ModuleList()
        for layer_id in range(args.gnn_layer_num):
            input_dim = self.gnn_input_dim if layer_id==0 else self.gnn_hidden_dim
            self.W.append(nn.Linear(input_dim, self.gnn_hidden_dim, True))
        
        self.transition = nn.Linear(self.gnn_hidden_dim, args.gnn_output_dim)
        self.K = K
        self.RL_K = args.RL_K
        self.RL_K_min = args.RL_K_min
        self.RL_K_max = args.RL_K_max
        self.RL_K_log = []
        self.RL_S = args.RL_S
        self.RL_R = args.RL_R
        self.RL_memory = []
        self.last_acc = 0.0
        self.RL_stop_flag = False

        self.gnn_proportion = args.gcn_proportion
        self.img_num = args.img_num

    def forward(self, encoder_outputs, syn_dep_adj, syn_dis_adj, search_flag, text_only=True):
        self.batch_size = encoder_outputs.shape[0]
        overall_max_len = encoder_outputs.shape[1]
        #print(syn_dep_adj.shape, syn_dis_adj.shape)
        bart_output = self.layernorm(encoder_outputs)
        if not text_only:
            image_feats = bart_output[:, :self.img_num+2, :]
            text_feats = bart_output[:, self.img_num+2:, :]
        else:
            text_feats = bart_output
        
        if self.abl_mode != 'type':
            syn_dep_adj_ = self.dep_imp_function(self.dep_embedding.weight, syn_dep_adj, overall_max_len, self.batch_size)
            dep_adj = syn_dep_adj_.float()

        if self.abl_mode != 'distance':
            if search_flag:
                syn_dis_adj_ = self.dis_imp_function_search(syn_dis_adj)
            else:
                syn_dis_adj_ = self.dis_imp_function(syn_dis_adj)
            dis_adj = syn_dis_adj_.float()
        
        if self.abl_mode == 'type':
            A = dis_adj
        elif self.abl_mode == 'distance':
            A = dep_adj
        else:
            A = torch.add(dep_adj, dis_adj)
        #print(A.shape, bart_output.shape)
        gnn_input = self.bart_drop(text_feats)
        gnn_output = gnn_input
        for layer_id in range(self.gnn_layer_num):
            gnn_output = self.W[layer_id](torch.matmul(A, gnn_output))
            gnn_output = F.relu(gnn_output)
            gnn_output = self.gnn_drop(gnn_output)
        output = F.relu(self.transition(gnn_output))
        if not text_only:
            output = torch.cat([image_feats, output], dim=1)
        
        #print(output.shape)
        return output
    
    def dis_imp_function(self, adj):
        adj_ = (1-torch.pow(adj/self.max_tree_dis, self.max_tree_dis))*torch.exp(-self.K*adj)
        return adj_
    
    def dis_imp_function_search(self, adj):
        adj_ = (1-torch.pow(adj/self.max_tree_dis, self.max_tree_dis))*torch.exp(-self.RL_K*adj)
        return adj_
    
    def reward_and_punishment(self, acc):
        if not self.RL_stop_flag:
            reward = -1 if acc <= self.last_acc else +1
            self.RL_memory.append(reward)
            self.last_acc = acc
            if len(self.RL_memory)>=self.RL_R and abs(sum(self.RL_memory[-self.RL_R:])) ==0 and len(self.RL_K_log)>100:
                self.RL_stop_flag = True
            self.RL_K = self.RL_K + self.RL_S if reward == +1 else self.RL_K - self.RL_S
            self.RL_K = self.RL_K_max if self.RL_K > self.RL_K_max else self.RL_K
            self.RL_K = self.RL_K_min if self.RL_K < self.RL_K_min else self.RL_K
            self.RL_K_log.append(self.RL_K)

class DEP_IMP(nn.Module):
    def __init__(self, att_dim):
        super(DEP_IMP, self).__init__()
        self.q = nn.Linear(att_dim, 1)
    def forward(self, input, syn_dep_adj, overall_max_len, batch_size):
        query = self.q(input).T
        att_adj = F.softmax(query, dim=-1)
        att_adj = att_adj.unsqueeze(0).repeat(batch_size, overall_max_len, 1)
        att_adj = torch.gather(att_adj, 2, syn_dep_adj)
        att_adj[syn_dep_adj == 0.] = 0.
        return att_adj
    
class Sim_GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(Sim_GCN, self).__init__()
        self.gcn_input_dim = input_dim
        self.gcn_output_dim = output_dim
        self.gcn_hidden_dim = output_dim
        self.gcn_layer_num = args.sim_gcn_layer_num

        self.W = nn.ModuleList()
        for layer_id in range(self.gcn_layer_num):
            input_dim = self.gcn_input_dim if layer_id==0 else self.gcn_hidden_dim
            self.W.append(nn.Linear(input_dim, self.gcn_hidden_dim, True))

        self.gcn_drop = nn.Dropout(args.sim_gcn_dropout)
        self.linear = nn.Linear(self.gcn_hidden_dim, self.gcn_output_dim)

    def forward(self, input_1, input_2, attention_mask, noun_mask=None):
        new_dependency_matrix=torch.zeros([input_2.shape[0],input_2.shape[1],input_2.shape[1]],dtype=torch.float).to(input_2.device)
        img_feature=input_2[:,:51,:]
        text_feature=input_2[:,51:,:]

        img_feature_extend=img_feature.unsqueeze(2).repeat(1,1,text_feature.shape[1],1)
        text_feature_extend=text_feature.unsqueeze(1).repeat(1,img_feature.shape[1],1,1)
        sim=torch.cosine_similarity(img_feature_extend,text_feature_extend,dim=-1)

        noun_mask=noun_mask[:,51:].unsqueeze(1).repeat(1,sim.shape[1],1)
        sim=sim*noun_mask.to(sim.device)
        new_dependency_matrix[:,:51,51:]=sim
        new_dependency_matrix[:,51:,:51]=torch.transpose(sim,1,2)

        text_feature_extend1 = text_feature.unsqueeze(1).repeat(1, text_feature.shape[1], 1, 1)
        text_feature_extend2 = text_feature.unsqueeze(2).repeat(1, 1, text_feature.shape[1], 1)
        text_sim = torch.cosine_similarity(text_feature_extend1, text_feature_extend2, dim=-1)
        new_dependency_matrix[:, 51:, 51:] = text_sim
        adj_matrix = new_dependency_matrix

        # embeddings_norm = F.normalize(input_2, p=2, dim=-1)
        # adj_matrix = torch.matmul(embeddings_norm, embeddings_norm.transpose(1, 2))
        # # adj_matrix = torch.cosine_similarity(embeddings_norm, embeddings_norm, dim=-1)
        # #print(adj_matrix, adj_matrix.shape, input_emb.shape)
        # expanded_mask = attention_mask.unsqueeze(-1)
        # adj_matrix = adj_matrix * expanded_mask * expanded_mask.transpose(1, 2)

        gcn_output = input_1
        for layer_id in range(self.gcn_layer_num):
            gcn_output = self.W[layer_id](torch.matmul(adj_matrix, gcn_output))
            gcn_output = F.relu(gcn_output)
            gcn_output = self.gcn_drop(gcn_output)

        gcn_output = F.relu(self.linear(gcn_output))
        return gcn_output