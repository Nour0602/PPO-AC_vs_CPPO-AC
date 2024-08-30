"""
Created on Sun Jan 30 23:24:42 2022

@author: Nour_El_Houda 
"""
import torch
import argparse
from aggr_mat import *
from utils import *
from Params import configs
from Inst_generator import uni_instance_gen

def test():  
    device = torch.device(configs.device)

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=15, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=15, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=15, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=15, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
    params = parser.parse_args()
    
    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = 1
    HIGH = 99
    SEED = params.seed
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m
    
    
    from reset_set_0 import SJSSP
    from PPO_jssp_multiInstances import PPO
    env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)
    #print(env.number_of_jobs)
    
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    path = './{}.pth'.format(str(15) + '_' + str(15) + '_' + str(1) + '_' + str(99))
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                              batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                              n_nodes=env.number_of_tasks,
                              device=device)
    
    adj, fea, candidate, mask = env.reset(uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=1, high=99))
    ep_reward = - env.max_endTime
    while True:
        fea_tensor = torch.from_numpy(fea).to(device)
        adj_tensor = torch.from_numpy(adj).to(device)
        candidate_tensor = torch.from_numpy(candidate).to(device)
        mask_tensor = torch.from_numpy(mask).to(device)
  

        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                                graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
            action = greedy_select_action(pi, candidate)

        adj, fea, reward, done, candidate, mask = env.step(action)
        ep_reward += reward

        if done:
            break
    
    return env.dur, env.m,env.opIDsOnMchs,env.mchsStartTimes,env.temp1, env.partial_sol_sequeence


