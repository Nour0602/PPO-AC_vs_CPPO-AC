import torch.nn as nn
from DNN_0 import MLPActor
from DNN_0 import MLPCritic
import torch.nn.functional as F
from GNN import GraphCNN
import torch


class ActorCriticCost(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCriticCost, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)
        self.critic_r = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

        self.critic_c= MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
       
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
  
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
      
        
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
    
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)
        
        mask_reshape = mask.reshape(candidate_scores.size())
     
        candidate_scores[mask_reshape] = float('-inf')
        
        pi = F.softmax(candidate_scores, dim=1)
        #print("pi is:", pi[0][0])
        v = self.critic_r(h_pooled)
        v_c = self.critic_c(h_pooled)
        #print("value of critic is (of forward actor_critic) :",   v)
        return pi, v, v_c
class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        # print("h_pooled")
        # print(h_pooled.size(1))
        # print("h_nodes")
        # print(h_nodes.size(0))
        #print("self.n_j",   self.n_j)
        # prepare policy feature: concat omega feature with global feature
        #print("candidate is:" , candidate)
        #print("first part: unsqueeze reseult:,",candidate.unsqueeze(-1) )
        #print("self.n_j:", self.n_j)
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        # print("dummy is:", dummy.size(0))
        # print("h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)):", h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).size(1))
        #print("dummy.size(0)", dummy.size(-2))
        #print("h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)):",  torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy).size())
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        #print("candidate_feature:", candidate_feature.size())
        
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        #print("h_pooled_repeated", h_pooled_repeated)

        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''

        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)
        #print("candidate_scores:", candidate_scores)

        #perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        #print("mask_ reshape of the forward actor_critic")
        #print(mask_reshape)
        candidate_scores[mask_reshape] = float('-inf')
        
        pi = F.softmax(candidate_scores, dim=1)
        #print("pi is:", pi[0][0])
        v = self.critic(h_pooled)
        
        #print("value of critic is (of forward actor_critic) :",   v)
        return pi, v

if __name__ == '__main__':
    print('Go home')