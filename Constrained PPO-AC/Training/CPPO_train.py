
"""
Created on Sun Jan 30 23:24:42 2022

@author: Nour_El_Houda HA
"""

import torch
import time
import numpy as np
import torch.nn as nn
from torch.nn.functional import softplus
from statistics import mean
from copy import deepcopy


from aggr_mat import *
from utils import eval_actions
from utils import select_action
from AC_DNN import ActorCriticCost
from Parameters import configs
from reset_set_env import SJSSP
from Inst_generator import uni_instance_gen
from generate_ref import test
from random import randrange
device = torch.device(configs.device)
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []
        self.dev_mb =[]
        self.lg=[]

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]
        del self.dev_mb[:]
        del self.lg[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCriticCost(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)
        
        self.penalty = torch.tensor(1.0, device=device, requires_grad=True).float()
        self.penalty_optimizer = torch.optim.Adam([self.penalty], lr=5e-1)
        self.penalty_scheduler = torch.optim.lr_scheduler.StepLR(self.penalty_optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool, cost_limit):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        costs_all_env = []

        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            costs =[]
            rewards = []
            discounted_reward = 0
            discounted_cost = 0

            # check the change for the consideration of cost memory:
            for reward, cost, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].dev_mb), reversed(memories[i].done_mb)):
                # print("reversed(memories[i].dev_mb):",memories[i].dev_mb,list(reversed(memories[i].dev_mb)))
                # print("rew,cost:", reward, cost,  is_terminal)
                if is_terminal:
                    discounted_reward = 0
                    discounted_cost = 0

                discounted_reward = reward + (self.gamma * discounted_reward)
                discounted_cost = cost + (self.gamma * discounted_cost)

                rewards.insert(0, discounted_reward)
                costs.insert(0, discounted_cost)

            # print("costs are:", costs, rewards)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)

            costs = torch.tensor(costs, dtype=torch.float).to(device)
            costs = (costs - costs.mean()) / (costs.std() + 1e-5)
            costs_all_env.append(costs)

            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Optimize policy for K epochs:
        penalty_list=[]
        for _ in range(self.k_epochs):
            
            penalty_loss_sum = 0
            #Update the penalty parameter first
            for i in range(len(memories)):
                penalty_loss = -self.penalty*(memories[i].dev_mb[-1] - cost_limit)
                penalty_loss_sum += penalty_loss
                # print("for penalty tracking:",memories[i].dev_mb[-1],cost_limit, self.penalty.item(), penalty_loss)
            
            mean_cost=0 
            for i in range(len(memories)):
                mean_cost += memories[i].dev_mb[-1]

            mean_cost = mean_cost/len(memories)
            print("mean cost is", mean_cost)
            if True: #mean_cost > cost_limit:
                # print(f"--- updating the {self.penalty.item() =} for {mean_cost =} vs {cost_limit =}")
                self.penalty_optimizer.zero_grad()
                penalty_loss_sum.mean().backward()
                self.penalty_optimizer.step()
                #  Positive part
                self.penalty.data.clamp_min_(0)
                # self.penalty = softplus(self.penalty)
                # self.penalty=penalty
                print(f" Lambda : {self.penalty.data.item()}, cost limit : {cost_limit}")
            # self.penalty_optimizer.zero_grad()
            # penalty_loss_sum.mean().backward()
            # self.penalty_optimizer.step()

            # penalty = self.penalty
            # p = softplus(penalty)
            # penalty_item = p.item()
            # penalty_list.append(penalty_item)
            penalty_list.append(self.penalty.item())

            loss_sum = 0

            vloss_sum = 0
            ploss_sum = 0

            v_c_loss_sum = 0
            
            for i in range(len(memories)):
                
                pis, vals, vals_c = self.policy(x=fea_mb_t_all_env[i].float(),
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())

                # print("rewards_all_env[i:],", rewards_all_env[i], costs_all_env[i])
                advantages = rewards_all_env[i] - vals.detach()
                advantages_c= costs_all_env[i] - vals_c.detach()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                surr1_c = ratios * advantages_c
               


                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                v_c_loss= self.V_loss_2(vals_c.squeeze(), costs_all_env[i])

                # penalty= self.penalty
                # p = softplus(penalty)
                # penalty_item = p.item()

                # p_loss = -(torch.min(surr1, surr2) - penalty_item*surr1_c)/(1+penalty_item)
                p_loss = -(torch.min(surr1, surr2) - self.penalty*surr1_c)/(1+self.penalty)

                ent_loss = -ent_loss.clone()
                loss = vloss_coef*(v_loss +v_c_loss) + ploss_coef*p_loss + entloss_coef*ent_loss

                loss_sum += loss

                v_c_loss_sum += v_c_loss
                vloss_sum += v_loss

                ploss_sum += p_loss 
    

            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
            # self.penalty_scheduler.step()

        return loss_sum.mean().item(), vloss_sum.mean().item(), ploss_sum.mean().item(),v_c_loss_sum.mean().item(),penalty_loss_sum.mean().item(), mean_cost,  penalty_list

def is_increasing(lst):
    return all(lst[i] < lst[i+1] for i in range(len(lst)-1))

def is_decreasing(lst):
    return all(lst[i] >= lst[i+1] for i in range(len(lst)-1))
def main():

    
    envs = [SJSSP(n_j=configs.n_j, n_m=configs.n_m) for _ in range(configs.num_envs)]
    
    
    
    data_generator = uni_instance_gen
   
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)    
    np.random.seed(configs.np_seed_train)
    
    add_poisson=1
  
    memories = [Memory() for _ in range(configs.num_envs)]
    
    ppo_1 = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
            n_j=envs[0].number_of_jobs+add_poisson,
            n_m=envs[0].number_of_machines, 
            num_layers=configs.num_layers,
            neighbor_pooling_type=configs.neighbor_pooling_type,
            input_dim=configs.input_dim,
            hidden_dim=configs.hidden_dim,
            num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
            num_mlp_layers_actor=configs.num_mlp_layers_actor,
            hidden_dim_actor=configs.hidden_dim_actor,
            num_mlp_layers_critic=configs.num_mlp_layers_critic,
            hidden_dim_critic=configs.hidden_dim_critic)
           
    g_pool_step_1=g_pool_cal(graph_pool_type=configs.graph_pool_type, #nb of nodes msut be generalized for all environments
                             batch_size=torch.Size([1, envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-2250,envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-2250]),
                             n_nodes=envs[0].number_of_tasks+add_poisson*envs[0].number_of_machines-2250,
                             device=device)
    
    # training loop
    data_v_loss=[]
    data_vc_loss=[]
    data_pen_loss=[]
    data_cost=[]
    data_pen_coef=[]
    lop=[]
    lo1=[]
    log =[]

    lg_coef=[]

    reward_list=[]
    ms_list=[]
    de_list=[]
    moving_avg = []
    moving_de =[]
    moving_ms=[] 
    moving_cost=[]
    
    delta=10
    window_size=6000
    i_update=0
    cost_limit=100
    # cost_limit=0
    cost_limit_list=[cost_limit]
    STOP=False
    while  i_update<=10000:
        # print("episode:", i_update)
        ep_rewards = [0 for _ in range(configs.num_envs)]
        ep_ms=[0 for _ in range(configs.num_envs)]
        ep_de=[0 for _ in range(configs.num_envs)]
       
        tuple_cmax=[0 for _ in range(configs.num_envs)] 

        adj_envs = []
        fea_envs = []
        candidate_envs = []
        mask_envs = []

        DEV=[]
        data=[]
        # np.random.seed(configs.np_seed_train)
        for i, env in enumerate(envs):
            dur,seq=data_generator(n_j=configs.n_j, n_m=configs.n_m, low=1, high=99)
            data_add=data_generator(n_j=add_poisson, n_m=configs.n_m, low=1, high=99)
            data1,data2,mchIds,start,end, sol_seq=test()
            tuple_cmax[i]=end
            data=(data1,data2)
            eliminated=sol_seq
           
            while (len(eliminated))>2250:
                eliminated.pop()

            row_eli=eliminated[-1]//configs.n_m
            col_eli=eliminated[-1]  % configs.n_m
            hour1=end[row_eli,col_eli]

            op=eliminated[-1]
            mch_op=np.take(data2,op)-1
            ind1=np.where(mchIds[mch_op]==op)[0][0]
            start_operation= start[mch_op,ind1]
            hour2=start_operation
            hour=int(np.random.uniform(hour2, hour1))

            adj, dev,fea, candidate,candidate_ch,mask, nb_nodes, temp1, opIDsOnMchs,mchsStartTimes = env.reset_add_eliminate(data,start,mchIds,add_poisson,data_add,[eliminated[i]for i in range(len(eliminated))],True,end,end) #

            adj_envs.append(adj)
            fea_envs.append(fea)
            candidate_envs.append(candidate_ch)
            mask_envs.append(mask)
            ep_rewards[i] = - env.initQuality

            
        while True:
            fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
            adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
            candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in candidate_envs]
            mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]
                       
            with torch.no_grad():
                action_envs = []
                a_idx_envs = []
                for i in range(configs.num_envs):
                    pi1, v,v_c = ppo_1.policy_old(x=fea_tensor_envs[i].float(),
                                           graph_pool=g_pool_step_1,
                                           padded_nei=None,
                                           adj=adj_tensor_envs[i],
                                           candidate=candidate_tensor_envs[i].type(torch.int64).unsqueeze(0),
                                           mask=mask_tensor_envs[i].unsqueeze(0))

                    action_idk, a_idx = select_action(pi1, candidate_envs[i], memories[i])
                    
                    key_listed=list(envs[i].idd.keys())
                    value_listed=list(envs[i].idd.values())
                    
                    #print("key_listed,value_listed at training:", key_listed,value_listed)
                    
                    position2=key_listed.index(action_idk)
                    action_idv=value_listed[position2]
                    

                    action_idk=np.int64(action_idk)
                    action_idv=np.int64(action_idv)
                    action_envs.append([action_idk,action_idv])
                    
                    a_idx_envs.append(a_idx)
            
            
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []
            
            # Saving episode data
            DEV=[]
            nb_job_de=[0,0]
            for i in range(configs.num_envs):
                memories[i].adj_mb.append(adj_tensor_envs[i])
                memories[i].fea_mb.append(fea_tensor_envs[i])
                memories[i].candidate_mb.append(candidate_tensor_envs[i])
                memories[i].mask_mb.append(mask_tensor_envs[i])
                memories[i].a_mb.append(a_idx_envs[i])

                adj,dev, fea, reward, done, candidate_ch, mask = envs[i].step(hour,configs.n_j,action_envs[i][0].item(),action_envs[i][1].item(),[eliminated[i]for i in range(len(eliminated))],add_poisson)
                        
                adj_envs.append(adj)
                DEV.append(dev) ##deviation matrix 
                fea_envs.append(fea)
                candidate_envs.append(candidate_ch)
                mask_envs.append(mask)
                
                ep_rewards[i] += reward
                sum1=dev[:,-1].sum()
                max_de=np.max(dev)
                ep_de[i]=sum1
                ep_ms[i]=envs[i].temp1.max()

                # for k in range(100):
                #         if envs[i].temp1[k][49]>tuple_cmax[i][k][49]:
                #             nb_job_de[i]+=1
                            # print("it is", nb_job_de,envs[i].temp1[k][14], tuple_cmax[i][k][14] )
                          
                # memories[i].dev_mb.append(nb_job_de[i]*10)


                # memories[i].dev_mb.append(np.max(dev))
                # memories[i].dev_mb.append(sum1)
                #to involve mkespan as cost

                #21/02/2024
                # memories[i].dev_mb.append(max_de)

                memories[i].dev_mb.append(sum1)
                memories[i].r_mb.append(reward)
                memories[i].done_mb.append(done)
                
               
            if envs[0].done():
                break
            
        for j in range(configs.num_envs):
            ep_rewards[j] -= envs[j].posRewards
        
        if (i_update>0) and (i_update%5==0):
            
            if is_increasing(data_pen_coef[-5:]):
                cost_limit+=100
              
            elif  is_decreasing(data_pen_coef[-5:]):
                cost_limit-=100
              
            cost_limit_list.append(cost_limit)
            
           
        
        loss, v_loss, p_loss, v_c_loss,penalty_loss, mean_cost,  penalty_list = ppo_1.update(memories, envs[0].nb_nodes, configs.graph_pool_type,cost_limit)

        for memory in memories:
            memory.clear_memory()

        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        mean_ms_all_env=sum(ep_ms) / len(ep_rewards)
        mean_de_all_env=sum(ep_de) / len(ep_rewards)

       

        reward_list.append(mean_rewards_all_env)
        ms_list.append(mean_ms_all_env)
        de_list.append(mean_de_all_env)
        data_v_loss.append(v_loss)
        lop.append(mean_cost)
        lg_coef.append(v_c_loss)
        data_pen_loss.append(penalty_loss)
        data_pen_coef.append(penalty_list[0])

        log.append([i_update, mean_rewards_all_env])
        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}\t '.format(i_update + 1, mean_rewards_all_env, v_loss))
        
        if i_update%1000==0:
            file_writing_log=open('./' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_log.write(str(log))

            file_writing_lg_loss = open(
            './' + 'lg_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_lg_loss.write(str(lg_coef)) 

            file_writing_cost_limit = open(
            './' + 'cost_limit_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_cost_limit.write(str(cost_limit_list)) 

            
            file_writing_pen_loss = open(
            './' + 'pen_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_pen_loss.write(str(data_pen_loss)) 

            file_writing_pen_co = open(
            './' + 'pen_co_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
            file_writing_pen_co.write(str(data_pen_coef)) 



        
        #Early stopping study
        if (i_update + 1) == window_size:

            torch.save(ppo_1.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_1' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))
            

            
           
            R_max= mean(reward_list[-window_size:])
            # sum_min=mean(sum_list[-window_size:])
            de_min=mean(de_list[-window_size:])
            ms_min=mean(ms_list[-window_size:])
            cost_mean=mean(lop[-window_size:])

            moving_avg.append(R_max)
            # moving_sum.append(sum_min)
            moving_de.append(de_min)
            moving_ms.append(ms_min)
            moving_cost.append(cost_mean)

        elif (i_update + 1) > window_size:
            R_i= mean(reward_list[-window_size:])
            # sum_min=mean(sum_list[-window_size:])
            de_min=mean(de_list[-window_size:])
            ms_min=mean(ms_list[-window_size:])
            cost_mean=mean(lop[-window_size:])

            moving_avg.append(R_i)
            # moving_sum.append(sum_min)
            moving_de.append(de_min)
            moving_ms.append(ms_min)
            moving_cost.append(cost_mean)
           
            
            if R_i> R_max:
               
                R_max=R_i  
          
                torch.save(ppo_1.policy.state_dict(), './{}.pth'.format(
                    str(configs.n_j) + '_1' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high)))

                file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_rew_window.write(str(moving_avg))
           
                
                file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_vloss.write(str(data_v_loss))

                file_writing_cost_window=open('./' + 'moving_cost_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_cost_window.write(str(moving_cost))
               
                file_writing_obj1 = open(
                './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj1.write(str(moving_ms))
            
                file_writing_obj2 = open(
                './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj2.write(str(moving_de)) 

                file_writing_lg_loss = open(
                './' + 'lg_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_lg_loss.write(str(lg_coef)) 

                file_writing_pen_loss = open(
                './' + 'pen_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_pen_loss.write(str(data_pen_loss)) 

                file_writing_pen_co = open(
                './' + 'pen_co_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_pen_co.write(str(data_pen_coef)) 


            else:
                if 100*abs((R_max - R_i)/R_max)>=delta:
                # if 100*((R_max - R_i)/R_max)>=delta:
                   
                    STOP=True
                    file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_rew_window.write(str(moving_avg))

                    file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_vloss.write(str(data_v_loss))

                    file_writing_cost_window=open('./' + 'moving_cost_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_cost_window.write(str(moving_cost))

                    file_writing_obj1 = open(
                    './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj1.write(str(moving_ms))       
                
                    file_writing_obj2 = open(
                    './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj2.write(str(moving_de))


                    file_writing_lg_loss = open(
                    './' + 'lg_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_lg_loss.write(str(lg_coef))
                    file_writing_pen_loss = open(
                    './' + 'pen_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_pen_loss.write(str(data_pen_loss)) 

                    file_writing_pen_co = open(
                    './' + 'pen_co_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_pen_co.write(str(data_pen_coef)) 

                else:
                    file_writing_rew_window=open('./'+'moving_rew_'+ str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_rew_window.write(str(moving_avg))


                    file_writing_vloss=open('./' + 'VLoss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_vloss.write(str(data_v_loss))

                    
                    file_writing_cost_window=open('./' + 'moving_cost_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_cost_window.write(str(moving_cost))

                    file_writing_obj1 = open(
                    './' + 'valiMSP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj1.write(str(moving_ms))       
                
                    file_writing_obj2 = open(
                    './' + 'valiDEVP_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_obj2.write(str(moving_de))

                    file_writing_lg_loss = open(
                    './' + 'lg_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_lg_loss.write(str(lg_coef))

                    file_writing_pen_loss = open(
                    './' + 'pen_loss_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_pen_loss.write(str(data_pen_loss)) 

                    file_writing_pen_co = open(
                    './' + 'pen_co_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                    file_writing_pen_co.write(str(data_pen_coef)) 
        
        i_update+=1

if __name__ == '__main__':
    total1 = time.time()
    main()
    total2 = time.time()
    print("Training time is:",total2 - total1)
